use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use crate::aec_trait::AecProcessor;
use crate::sync_buffer::AecSyncManager;

/// A synchronized AEC wrapper that handles timing alignment
pub struct SynchronizedAec {
    inner: Box<dyn AecProcessor>,
    sync_manager: Arc<Mutex<AecSyncManager>>,
    sample_rate: u32,
    lookahead_ms: u32,
}

impl SynchronizedAec {
    pub fn new(inner: Box<dyn AecProcessor>, sample_rate: u32) -> Self {
        let sync_manager = Arc::new(Mutex::new(
            AecSyncManager::new(sample_rate, 1000, 50) // 1s buffer, 50ms estimated delay
        ));
        
        Self {
            inner,
            sync_manager,
            sample_rate,
            lookahead_ms: 100, // 100ms lookahead as requested
        }
    }
    
    /// Process a duplex frame with proper synchronization
    pub fn process_duplex_frame(&mut self, 
                               render_samples: &[f32],  // Audio going to speakers
                               capture_samples: &[f32], // Audio from microphone
                               output: &mut [f32],      // Processed capture (echo cancelled)
                               timestamp: Instant) -> Result<(), Box<dyn std::error::Error>> {
        // Store render samples with timestamp
        {
            let mut sync = self.sync_manager.lock().unwrap();
            sync.add_render_samples(render_samples, timestamp);
        }
        
        // Get synchronized render samples for this capture frame
        let synchronized_render = {
            let sync = self.sync_manager.lock().unwrap();
            sync.get_synchronized_render(timestamp, capture_samples.len(), self.lookahead_ms)
        };
        
        // Process render with the synchronized samples
        self.inner.process_render(&synchronized_render)?;
        
        // Process capture
        self.inner.process_capture(capture_samples, output)?;
        
        Ok(())
    }
    
    /// Reset the AEC and clear sync buffers
    pub fn reset(&mut self) {
        self.inner.reset();
        let mut sync = self.sync_manager.lock().unwrap();
        sync.clear();
    }
}

/// Alternative approach: Ring buffer based synchronization
pub struct RingBufferSync {
    render_history: VecDeque<f32>,
    history_size: usize,
    delay_samples: usize,
}

impl RingBufferSync {
    pub fn new(sample_rate: u32, max_delay_ms: u32, expected_delay_ms: u32) -> Self {
        let history_size = (sample_rate as usize * max_delay_ms as usize) / 1000;
        let delay_samples = (sample_rate as usize * expected_delay_ms as usize) / 1000;
        
        Self {
            render_history: VecDeque::with_capacity(history_size),
            history_size,
            delay_samples,
        }
    }
    
    /// Add render samples to history
    pub fn push_render(&mut self, samples: &[f32]) {
        self.render_history.extend(samples);
        
        // Keep buffer size bounded
        while self.render_history.len() > self.history_size {
            self.render_history.pop_front();
        }
    }
    
    /// Get render samples from the past that correspond to current capture
    pub fn get_delayed_render(&self, num_samples: usize) -> Vec<f32> {
        let start_idx = self.render_history.len().saturating_sub(self.delay_samples + num_samples);
        let end_idx = start_idx + num_samples;
        
        if end_idx <= self.render_history.len() {
            self.render_history.range(start_idx..end_idx).copied().collect()
        } else {
            // Not enough history, pad with zeros
            let mut result = Vec::with_capacity(num_samples);
            for i in start_idx..self.render_history.len().min(end_idx) {
                if i < self.render_history.len() {
                    result.push(self.render_history[i]);
                }
            }
            result.resize(num_samples, 0.0);
            result
        }
    }

    /// Fill the provided buffer with delayed render samples without allocating
    pub fn fill_delayed_render(&self, dst: &mut [f32]) {
        let num_samples = dst.len();
        let start_idx = self
            .render_history
            .len()
            .saturating_sub(self.delay_samples + num_samples);
        let end_idx = start_idx + num_samples;

        let mut wrote = 0usize;
        if end_idx <= self.render_history.len() {
            for (i, s) in self
                .render_history
                .range(start_idx..end_idx)
                .copied()
                .enumerate()
            {
                dst[i] = s;
                wrote += 1;
            }
        } else {
            // Not enough history, copy what we have
            let available_end = self.render_history.len().min(end_idx);
            let available_start = start_idx.min(self.render_history.len());
            for i in available_start..available_end {
                if i < self.render_history.len() && wrote < num_samples {
                    dst[wrote] = self.render_history[i];
                    wrote += 1;
                } else {
                    break;
                }
            }
        }

        // Pad remainder with zeros
        if wrote < num_samples {
            for v in &mut dst[wrote..] {
                *v = 0.0;
            }
        }
    }
    
    pub fn clear(&mut self) {
        self.render_history.clear();
    }
}
