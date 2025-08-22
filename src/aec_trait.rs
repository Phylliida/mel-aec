use std::error::Error;

/// Common trait for AEC implementations
pub trait AecProcessor: Send + Sync {
    /// Process render (output/playback) samples
    fn process_render(&mut self, samples: &[f32]) -> Result<(), Box<dyn Error>>;
    
    /// Process capture (input/microphone) samples with AEC
    fn process_capture(&mut self, input: &[f32], output: &mut [f32]) -> Result<usize, Box<dyn Error>>;
    
    /// Reset the AEC state
    fn reset(&mut self);
}

/// Enum to select AEC implementation
#[derive(Debug, Clone, Copy)]
pub enum AecType {
    WebRtc,
    Fdaf,
}

