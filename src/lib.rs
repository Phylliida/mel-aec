use pyo3::prelude::*;

mod audio_engine;
mod aec;
mod buffer;
mod stream;

use audio_engine::AudioEngine;
use stream::{DuplexStream, StreamConfig};

/// Main Python module for the audio AEC library
#[pymodule]
fn audio_aec(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AudioEngine>()?;
    m.add_class::<DuplexStream>()?;
    m.add_class::<StreamConfig>()?;
    
    Ok(())
}