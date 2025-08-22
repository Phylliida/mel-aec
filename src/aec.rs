use std::sync::{Arc, Mutex};
use std::error::Error;
use webrtc_audio_processing::{
    Config, EchoCancellation, EchoCancellationSuppressionLevel,
    GainControl, GainControlMode, InitializationConfig, 
    NoiseSuppression, NoiseSuppressionLevel, Processor
};
use crate::aec_trait;

/// WebRTC-based AEC processor configuration
#[derive(Debug, Clone)]
pub struct AecConfig {
    pub sample_rate: u32,
    pub channels: u32,
    pub enable_aec: bool,
    pub enable_agc: bool,
    pub enable_ns: bool,
}

impl Default for AecConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            channels: 1,
            enable_aec: true,
            enable_agc: false,
            enable_ns: false,
        }
    }
}

/// WebRTC-based AEC processor
pub struct AecProcessor {
    processor: Arc<Mutex<Processor>>,
    config: AecConfig,
    // Buffers for processing frames
    render_buffer: Arc<Mutex<Vec<f32>>>,
    capture_buffer: Arc<Mutex<Vec<f32>>>,
    frame_size: usize,
}

impl AecProcessor {
    /// Create a new AEC processor
    pub fn new(config: AecConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // WebRTC requires specific sample rates
        let sample_rate = match config.sample_rate {
            8000 | 16000 | 32000 | 48000 => config.sample_rate,
            _ => return Err("Unsupported sample rate. Use 8000, 16000, 32000, or 48000 Hz".into()),
        };

        // Calculate frame size for 10ms based on sample rate
        // WebRTC processes audio in 10ms frames
        let frame_size = (sample_rate / 100) as usize;  // samples per 10ms

        // Create initialization config
        let mut init_config = InitializationConfig::default();
        init_config.num_capture_channels = config.channels as i32;
        init_config.num_render_channels = config.channels as i32;
        // Set sample rate if the field exists
        // init_config.sample_rate_hz = sample_rate as i32;

        // Create processor
        let mut processor = Processor::new(&init_config)?;

        // Configure echo cancellation
        let echo_cancellation = if config.enable_aec {
            Some(EchoCancellation {
                suppression_level: EchoCancellationSuppressionLevel::High,
                enable_delay_agnostic: true,  // Handle variable delays automatically
                enable_extended_filter: true,  // Better echo suppression
                stream_delay_ms: Some(50),  // Typical delay for macOS audio (can be tuned)
            })
        } else {
            None
        };

        // Configure gain control
        let gain_control = if config.enable_agc {
            Some(GainControl {
                mode: GainControlMode::AdaptiveDigital,
                target_level_dbfs: 3,
                compression_gain_db: 9,
                enable_limiter: true,
            })
        } else {
            None
        };

        // Configure noise suppression
        let noise_suppression = if config.enable_ns {
            Some(NoiseSuppression {
                suppression_level: NoiseSuppressionLevel::High,
            })
        } else {
            None
        };

        // Apply configuration
        let config_to_apply = Config {
            echo_cancellation,
            gain_control,
            noise_suppression,
            ..Config::default()
        };

        processor.set_config(config_to_apply);

        // Initialize buffers
        let render_buffer = vec![0.0f32; frame_size];
        let capture_buffer = vec![0.0f32; frame_size];

        Ok(Self {
            processor: Arc::new(Mutex::new(processor)),
            config,
            render_buffer: Arc::new(Mutex::new(render_buffer)),
            capture_buffer: Arc::new(Mutex::new(capture_buffer)),
            frame_size,
        })
    }

    /// Process render (output/playback) samples
    pub fn process_render(&mut self, samples: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        let mut render_buffer = self.render_buffer.lock().unwrap();
        let mut processor = self.processor.lock().unwrap();
        
        // Process complete frames
        for chunk in samples.chunks_exact(self.frame_size) {
            // Copy chunk to buffer
            render_buffer[..chunk.len()].copy_from_slice(chunk);
            
            // Process render stream (doesn't modify the buffer)
            processor.process_render_frame(&mut render_buffer)?;
        }
        
        Ok(())
    }

    /// Process capture (input/microphone) samples with AEC
    pub fn process_capture(&mut self, 
                          input: &[f32], 
                          output: &mut [f32]) -> Result<usize, Box<dyn std::error::Error>> {
        let mut capture_buffer = self.capture_buffer.lock().unwrap();
        let mut processor = self.processor.lock().unwrap();
        
        // Ensure output buffer is large enough
        if output.len() < input.len() {
            return Err("Output buffer too small".into());
        }
        
        let mut processed = 0;
        
        // Process complete frames
        for (chunk_idx, chunk) in input.chunks_exact(self.frame_size).enumerate() {
            // Copy chunk to buffer
            capture_buffer[..chunk.len()].copy_from_slice(chunk);
            
            // Process capture stream with AEC (modifies the buffer)
            processor.process_capture_frame(&mut capture_buffer)?;
            
            // Copy processed audio to output
            let out_start = chunk_idx * self.frame_size;
            output[out_start..out_start + self.frame_size]
                .copy_from_slice(&capture_buffer);
            
            processed += self.frame_size;
        }
        
        // Copy any remaining samples (partial frame) without processing
        let remaining = input.len() - processed;
        if remaining > 0 {
            output[processed..processed + remaining]
                .copy_from_slice(&input[processed..]);
            processed += remaining;
        }
        
        Ok(processed)
    }

    /// Reset the AEC state
    pub fn reset(&mut self) {
        // WebRTC processor doesn't expose a reset method
        // The best we can do is clear our buffers
        let mut render_buffer = self.render_buffer.lock().unwrap();
        let mut capture_buffer = self.capture_buffer.lock().unwrap();
        
        render_buffer.fill(0.0);
        capture_buffer.fill(0.0);
    }
}

impl aec_trait::AecProcessor for AecProcessor {
    fn process_render(&mut self, samples: &[f32]) -> Result<(), Box<dyn Error>> {
        AecProcessor::process_render(self, samples)
    }
    
    fn process_capture(&mut self, input: &[f32], output: &mut [f32]) -> Result<usize, Box<dyn Error>> {
        AecProcessor::process_capture(self, input, output)
    }
    
    fn reset(&mut self) {
        AecProcessor::reset(self);
    }
}