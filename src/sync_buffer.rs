use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// A sample with timestamp information
#[derive(Clone, Debug)]
struct TimestampedSample {
    sample: f32,
    timestamp: Instant,
}

/// Buffer that maintains samples with timestamps for synchronization
pub struct SyncBuffer {
    samples: VecDeque<TimestampedSample>,
    sample_rate: u32,
    max_duration: Duration,
}

impl SyncBuffer {
    pub fn new(sample_rate: u32, max_duration_ms: u32) -> Self {
        Self {
            samples: VecDeque::new(),
            sample_rate,
            max_duration: Duration::from_millis(max_duration_ms as u64),
        }
    }
    
    /// Add samples with current timestamp
    pub fn push_samples(&mut self, samples: &[f32], timestamp: Instant) {
        // Add new samples
        for (i, &sample) in samples.iter().enumerate() {
            // Calculate precise timestamp for each sample
            let sample_offset = Duration::from_secs_f64(i as f64 / self.sample_rate as f64);
            self.samples.push_back(TimestampedSample {
                sample,
                timestamp: timestamp + sample_offset,
            });
        }
        
        // Remove old samples beyond max duration
        // We only remove samples if we have a reference point
        while let Some(front) = self.samples.front() {
            // Check if the front sample is older than max_duration
            if front.timestamp + self.max_duration < timestamp {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }
    
    /// Get samples that were active at a specific time, with lookahead
    pub fn get_samples_at_time(&self, target_time: Instant, num_samples: usize, lookahead_ms: u32) -> Vec<f32> {
        let lookahead = Duration::from_millis(lookahead_ms as u64);
        let end_time = target_time + lookahead;
        
        let mut result = Vec::with_capacity(num_samples);
        
        // Find samples in the time window
        for timestamped in &self.samples {
            if timestamped.timestamp >= target_time && timestamped.timestamp < end_time {
                result.push(timestamped.sample);
                if result.len() >= num_samples {
                    break;
                }
            }
        }
        
        // Pad with zeros if not enough samples
        result.resize(num_samples, 0.0);
        result
    }
    
    /// Clear all samples
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

/// Synchronization manager for AEC
pub struct AecSyncManager {
    render_buffer: SyncBuffer,
    capture_start_time: Option<Instant>,
    render_start_time: Option<Instant>,
    sample_rate: u32,
    sync_delay_ms: u32,  // Estimated system delay
}

impl AecSyncManager {
    pub fn new(sample_rate: u32, buffer_duration_ms: u32, sync_delay_ms: u32) -> Self {
        Self {
            render_buffer: SyncBuffer::new(sample_rate, buffer_duration_ms),
            capture_start_time: None,
            render_start_time: None,
            sample_rate,
            sync_delay_ms,
        }
    }
    
    /// Called when render stream starts
    pub fn set_render_start(&mut self, time: Instant) {
        self.render_start_time = Some(time);
    }
    
    /// Called when capture stream starts
    pub fn set_capture_start(&mut self, time: Instant) {
        self.capture_start_time = Some(time);
    }
    
    /// Add render samples (output to speakers)
    pub fn add_render_samples(&mut self, samples: &[f32], timestamp: Instant) {
        self.render_buffer.push_samples(samples, timestamp);
    }
    
    /// Get synchronized render samples for a capture frame
    pub fn get_synchronized_render(&self, 
                                   capture_timestamp: Instant, 
                                   num_samples: usize,
                                   lookahead_ms: u32) -> Vec<f32> {
        // Adjust for estimated system delay by going back in time
        let delay = Duration::from_millis(self.sync_delay_ms as u64);
        let adjusted_time = capture_timestamp.checked_sub(delay).unwrap_or(capture_timestamp);
        
        self.render_buffer.get_samples_at_time(adjusted_time, num_samples, lookahead_ms)
    }
    
    /// Clear all buffers
    pub fn clear(&mut self) {
        self.render_buffer.clear();
    }
}
