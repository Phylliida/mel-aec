use pyo3::prelude::*;
use portaudio as pa;
use std::collections::HashMap;

use crate::stream::{DuplexStream, StreamConfig};

/// Main audio engine that manages devices and streams
#[pyclass]
pub struct AudioEngine {
    streams: HashMap<String, DuplexStream>,
    pa: pa::PortAudio,
}

#[pymethods]
impl AudioEngine {
    #[new]
    fn new() -> PyResult<Self> {
        let pa = pa::PortAudio::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to initialize PortAudio: {}", e)
            )
        })?;
        
        Ok(Self {
            streams: HashMap::new(),
            pa,
        })
    }
    
    /// List available audio devices
    fn list_devices(&self) -> PyResult<Vec<(String, String, bool, bool)>> {
        let mut devices = Vec::new();
        
        // Get default devices for reference
        let default_input = self.pa.default_input_device().ok();
        let default_output = self.pa.default_output_device().ok();
        
        // Iterate through all devices
        for device in self.pa.devices().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to enumerate devices: {}", e)
            )
        })? {
            let (idx, info) = device.map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to get device info: {}", e)
                )
            })?;
            
            let name = info.name.clone();
            let is_input = info.max_input_channels > 0;
            let is_output = info.max_output_channels > 0;
            
            let device_type = if Some(idx) == default_input && Some(idx) == default_output {
                "default_duplex"
            } else if Some(idx) == default_input {
                "default_input"
            } else if Some(idx) == default_output {
                "default_output"
            } else if is_input && is_output {
                "duplex"
            } else if is_input {
                "input"
            } else {
                "output"
            };
            
            devices.push((name.to_string(), device_type.to_string(), is_input, is_output));
        }
        
        Ok(devices)
    }
    
    /// Get supported sample rates for the default devices
    fn get_supported_configs(&self) -> PyResult<Vec<(u32, u16)>> {
        let mut configs = Vec::new();
        
        // Get default output device
        let device = self.pa.default_output_device().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("No default output device: {}", e)
            )
        })?;
        
        let info = self.pa.device_info(device).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to get device info: {}", e)
            )
        })?;
        
        // Test common sample rates
        let sample_rates = vec![8000, 16000, 22050, 44100, 48000];
        
        for rate in sample_rates {
            // Check if this sample rate is supported
            let params = pa::StreamParameters::<f32>::new(
                device,
                info.max_output_channels.min(2),  // Up to stereo
                true,
                info.default_low_output_latency,
            );
            
            if self.pa.is_output_format_supported(params, rate as f64).is_ok() {
                configs.push((rate, 1));  // Mono
                if info.max_output_channels >= 2 {
                    configs.push((rate, 2));  // Stereo
                }
            }
        }
        
        Ok(configs)
    }
    
    /// Create a new duplex stream
    fn create_stream(&mut self, name: String, config: StreamConfig) -> PyResult<()> {
        if self.streams.contains_key(&name) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Stream '{}' already exists", name)
            ));
        }
        
        let stream = DuplexStream::new(config)?;
        self.streams.insert(name, stream);
        Ok(())
    }
    
    /// Start a stream by name
    fn start_stream(&mut self, name: String) -> PyResult<()> {
        let stream = self.streams.get_mut(&name)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Stream '{}' not found", name)
            ))?;
        stream.start()
    }
    
    /// Stop a stream by name
    fn stop_stream(&mut self, name: String) -> PyResult<()> {
        let stream = self.streams.get_mut(&name)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Stream '{}' not found", name)
            ))?;
        stream.stop()
    }
    
    /// Write audio data to a stream
    fn write_output(&mut self, name: String, data: &[u8]) -> PyResult<usize> {
        let stream = self.streams.get(&name)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Stream '{}' not found", name)
            ))?;
        stream.write_output(data)
    }
    
    /// Read audio data from a stream
    fn read_input(&mut self, name: String, max_samples: usize) -> PyResult<Vec<u8>> {
        let stream = self.streams.get(&name)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Stream '{}' not found", name)
            ))?;
        stream.read_input(max_samples)
    }
    
    /// Get playback position for a stream
    fn get_playback_position(&self, name: String) -> PyResult<f64> {
        let stream = self.streams.get(&name)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Stream '{}' not found", name)
            ))?;
        Ok(stream.get_playback_position())
    }
    
    /// Get buffered samples for a stream
    fn get_output_buffer_size(&self, name: String) -> PyResult<usize> {
        let stream = self.streams.get(&name)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Stream '{}' not found", name)
            ))?;
        Ok(stream.get_output_buffer_size())
    }
    
    /// Interrupt output for a stream
    fn interrupt_output(&self, name: String) -> PyResult<()> {
        let stream = self.streams.get(&name)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Stream '{}' not found", name)
            ))?;
        stream.interrupt_output();
        Ok(())
    }
    
    /// Remove a stream
    fn remove_stream(&mut self, name: String) -> PyResult<()> {
        self.streams.remove(&name)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Stream '{}' not found", name)
            ))
            .map(|_| ())
    }
    
    /// Get info about host audio capabilities
    fn get_host_info(&self) -> PyResult<HashMap<String, String>> {
        let mut info = HashMap::new();
        
        // Get PortAudio version
        let version = self.pa.version();
        info.insert("backend".to_string(), "PortAudio".to_string());
        info.insert("version".to_string(), format!("{}", version));
        if let Ok(version_text) = self.pa.version_text() {
            info.insert("version_text".to_string(), version_text.to_string());
        }
        
        // Get host API info
        if let Ok(host_count) = self.pa.host_api_count() {
            info.insert("host_api_count".to_string(), format!("{}", host_count));
            
            if let Ok(default_host) = self.pa.default_host_api() {
                if let Some(host_info) = self.pa.host_api_info(default_host) {
                    info.insert("default_host_api".to_string(), host_info.name.to_string());
                }
            }
        }
        
        // Get default devices
        if let Ok(device) = self.pa.default_input_device() {
            if let Ok(device_info) = self.pa.device_info(device) {
                info.insert("default_input".to_string(), device_info.name.to_string());
            }
        }
        
        if let Ok(device) = self.pa.default_output_device() {
            if let Ok(device_info) = self.pa.device_info(device) {
                info.insert("default_output".to_string(), device_info.name.to_string());
            }
        }
        
        Ok(info)
    }
}