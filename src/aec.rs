use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use anyhow::Result;

/// Configuration for the acoustic echo canceller
#[derive(Debug, Clone)]
pub struct AecConfig {
    pub filter_length: usize,  // Typically 2048-4096 for 100-200ms tail
    pub frame_size: usize,      // Must match audio frame size
    pub sample_rate: u32,
}

impl Default for AecConfig {
    fn default() -> Self {
        Self {
            filter_length: 2048,
            frame_size: 256,
            sample_rate: 16000,
        }
    }
}

/// Simple adaptive echo canceller using NLMS algorithm
/// This is a basic implementation - for production use, consider WebRTC or SpeexDSP
pub struct AecProcessor {
    config: AecConfig,
    
    // Adaptive filter coefficients
    filter_coeffs: Arc<Mutex<Vec<f32>>>,
    
    // Reference signal buffer (playback)
    reference_buffer: Arc<Mutex<VecDeque<f32>>>,
    
    // Alignment buffers for precise timing
    playback_buffer: Arc<Mutex<Vec<f32>>>,
    capture_buffer: Arc<Mutex<Vec<f32>>>,
    
    // NLMS parameters
    step_size: f32,
    regularization: f32,
    
    // Timing information
    playback_delay_samples: Arc<Mutex<usize>>,
}

impl AecProcessor {
    pub fn new(config: AecConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            filter_coeffs: Arc::new(Mutex::new(vec![0.0; config.filter_length])),
            reference_buffer: Arc::new(Mutex::new(VecDeque::with_capacity(config.filter_length))),
            playback_buffer: Arc::new(Mutex::new(Vec::with_capacity(config.frame_size * 4))),
            capture_buffer: Arc::new(Mutex::new(Vec::with_capacity(config.frame_size * 4))),
            step_size: 0.1,  // NLMS step size
            regularization: 0.001,  // Small value to prevent division by zero
            playback_delay_samples: Arc::new(Mutex::new(0)),
        })
    }
    
    /// Feed playback (reference) signal to the echo canceller
    pub fn feed_playback(&self, samples: &[f32]) {
        let mut buffer = self.playback_buffer.lock().unwrap();
        buffer.extend_from_slice(samples);
        
        // Add to reference buffer for echo cancellation
        let mut ref_buffer = self.reference_buffer.lock().unwrap();
        
        // Process complete frames
        while buffer.len() >= self.config.frame_size {
            for &sample in &buffer[..self.config.frame_size] {
                ref_buffer.push_back(sample);
                if ref_buffer.len() > self.config.filter_length {
                    ref_buffer.pop_front();
                }
            }
            buffer.drain(..self.config.frame_size);
        }
    }
    
    /// Process captured audio to remove echo using NLMS algorithm
    pub fn process_capture(&self, input: &[f32], output: &mut [f32]) -> Result<usize> {
        if input.len() != output.len() {
            return Err(anyhow::anyhow!("Input and output buffers must be same size"));
        }
        
        let mut capture_buffer = self.capture_buffer.lock().unwrap();
        capture_buffer.extend_from_slice(input);
        
        let mut processed = 0;
        let mut out_offset = 0;
        
        while capture_buffer.len() >= self.config.frame_size && out_offset + self.config.frame_size <= output.len() {
            let ref_buffer = self.reference_buffer.lock().unwrap();
            let mut coeffs = self.filter_coeffs.lock().unwrap();
            
            // Process each sample in the frame
            for i in 0..self.config.frame_size {
                if out_offset + i >= output.len() {
                    break;
                }
                
                let mic_signal = capture_buffer[i];
                
                // Compute echo estimate using adaptive filter
                let mut echo_estimate = 0.0;
                let ref_vec: Vec<f32> = ref_buffer.iter().cloned().collect();
                
                for (j, &coeff) in coeffs.iter().enumerate() {
                    if j < ref_vec.len() {
                        echo_estimate += coeff * ref_vec[ref_vec.len() - 1 - j];
                    }
                }
                
                // Compute error (echo-cancelled signal)
                let error = mic_signal - echo_estimate;
                output[out_offset + i] = error;
                
                // Update filter coefficients using NLMS
                let mut power = 0.0;
                for j in 0..coeffs.len().min(ref_vec.len()) {
                    power += ref_vec[ref_vec.len() - 1 - j] * ref_vec[ref_vec.len() - 1 - j];
                }
                power += self.regularization;
                
                let update_factor = self.step_size * error / power;
                for j in 0..coeffs.len().min(ref_vec.len()) {
                    coeffs[j] += update_factor * ref_vec[ref_vec.len() - 1 - j];
                }
            }
            
            capture_buffer.drain(..self.config.frame_size);
            out_offset += self.config.frame_size;
            processed += self.config.frame_size;
        }
        
        Ok(processed)
    }
    
    /// Set the delay between playback and capture in samples
    pub fn set_delay(&self, delay_samples: usize) {
        *self.playback_delay_samples.lock().unwrap() = delay_samples;
    }
    
    /// Reset the echo canceller state
    pub fn reset(&self) {
        *self.filter_coeffs.lock().unwrap() = vec![0.0; self.config.filter_length];
        self.reference_buffer.lock().unwrap().clear();
        self.playback_buffer.lock().unwrap().clear();
        self.capture_buffer.lock().unwrap().clear();
    }
}