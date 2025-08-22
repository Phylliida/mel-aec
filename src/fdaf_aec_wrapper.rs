use crate::fdaf_aec::FdafAec;
use crate::aec_trait::AecProcessor;
use std::collections::VecDeque;
use std::error::Error;

/// Wrapper around FdafAec to implement the AecProcessor trait
pub struct FdafAecWrapper {
    inner: FdafAec,
    render_buffer: VecDeque<f32>,
    frame_size: usize,
    sample_rate: u32,
}

impl FdafAecWrapper {
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self, Box<dyn Error>> {
        if channels != 1 {
            return Err("FDAF AEC currently only supports mono audio".into());
        }
        
        // Choose FFT size based on sample rate to get ~64ms filter length
        let fft_size = match sample_rate {
            8000 => 512,   // 64ms filter
            16000 => 1024, // 64ms filter
            32000 => 2048, // 64ms filter
            48000 => 4096, // 85ms filter
            _ => return Err(format!("Unsupported sample rate: {}", sample_rate).into()),
        };
        
        let frame_size = fft_size / 2;
        let step_size = 0.8; // More aggressive learning rate for faster convergence
        
        let inner = FdafAec::new(fft_size, step_size);
        
        Ok(Self {
            inner,
            render_buffer: VecDeque::with_capacity(fft_size * 2),
            frame_size,
            sample_rate,
        })
    }
}

impl AecProcessor for FdafAecWrapper {
    fn process_render(&mut self, samples: &[f32]) -> Result<(), Box<dyn Error>> {
        // Buffer the render samples for later use in process_capture
        self.render_buffer.extend(samples);
        
        // Keep buffer size reasonable (max 2 seconds)
        let max_samples = self.sample_rate as usize * 2;
        while self.render_buffer.len() > max_samples {
            self.render_buffer.pop_front();
        }
        
        Ok(())
    }
    
    fn process_capture(&mut self, input: &[f32], output: &mut [f32]) -> Result<usize, Box<dyn Error>> {
        if output.len() < input.len() {
            return Err("Output buffer too small".into());
        }
        
        let mut processed = 0;
        
        // Process in frame_size chunks
        for chunk in input.chunks(self.frame_size) {
            if chunk.len() == self.frame_size {
                // Get corresponding render samples
                let render_samples: Vec<f32> = if self.render_buffer.len() >= self.frame_size {
                    // Drain the exact amount we need
                    self.render_buffer.drain(..self.frame_size).collect()
                } else {
                    // Not enough render samples, use silence
                    vec![0.0; self.frame_size]
                };
                
                // Process with FDAF
                let cancelled = self.inner.process(&render_samples, chunk);
                
                // Copy to output
                output[processed..processed + self.frame_size].copy_from_slice(&cancelled);
                processed += self.frame_size;
            } else {
                // Partial frame - copy without processing
                let remaining = chunk.len();
                output[processed..processed + remaining].copy_from_slice(chunk);
                processed += remaining;
            }
        }
        
        Ok(processed)
    }
    
    fn reset(&mut self) {
        // Recreate the FDAF instance to reset all state
        let fft_size = self.frame_size * 2;
        let step_size = 0.8;
        self.inner = FdafAec::new(fft_size, step_size);
        self.render_buffer.clear();
    }
}
