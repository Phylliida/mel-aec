use pyo3::prelude::*;
use pyo3::types::PyBytes;
use portaudio as pa;
use std::collections::HashMap;

#[cfg(unix)]
use std::sync::{Mutex, OnceLock};

use crate::stream::{DuplexStream, StreamConfig};

const COMMON_SAMPLE_RATES: [f64; 13] = [
    8000.0,
    11025.0,
    12000.0,
    16000.0,
    22050.0,
    24000.0,
    32000.0,
    44100.0,
    48000.0,
    88200.0,
    96000.0,
    176400.0,
    192000.0,
];

/// Main audio engine that manages devices and streams
#[pyclass]
pub struct AudioEngine {
    streams: HashMap<String, DuplexStream>,
    pa: pa::PortAudio,
}

impl AudioEngine {
    fn supported_sample_rates(
        &self,
        device: pa::DeviceIndex,
        max_channels: i32,
        latency: pa::Time,
        default_rate: f64,
        is_input: bool,
    ) -> Vec<f64> {
        if max_channels <= 0 {
            return Vec::new();
        }

        let mut candidates: Vec<f64> = COMMON_SAMPLE_RATES.iter().copied().collect();
        if default_rate.is_finite() && default_rate > 0.0 {
            candidates.push(default_rate);
        }
        candidates.retain(|rate| rate.is_finite() && *rate > 0.0);
        candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        candidates.dedup();

        candidates
            .into_iter()
            .filter(|&rate| {
                self.is_sample_rate_supported(device, max_channels, latency, rate, is_input)
            })
            .collect()
    }

    fn is_sample_rate_supported(
        &self,
        device: pa::DeviceIndex,
        max_channels: i32,
        latency: pa::Time,
        rate: f64,
        is_input: bool,
    ) -> bool {
        let max_channels = if max_channels < 0 {
            0
        } else {
            max_channels as usize
        };

        if max_channels == 0 {
            return false;
        }

        let mut channel_counts: Vec<usize> = (1..=max_channels.min(8)).collect();
        if max_channels > 8 {
            channel_counts.push(max_channels);
        }
        channel_counts.sort_unstable();
        channel_counts.dedup();

        for channels in channel_counts {
            let params =
                pa::StreamParameters::<f32>::new(device, channels as i32, true, latency);
            let result = suppress_portaudio_errors(|| {
                if is_input {
                    self.pa.is_input_format_supported(params, rate)
                } else {
                    self.pa.is_output_format_supported(params, rate)
                }
            });

            if result.is_ok() {
                return true;
            }
        }

        false
    }
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
    fn list_devices(
        &self,
    ) -> PyResult<Vec<(String, String, bool, bool, Vec<f64>, Vec<f64>)>> {
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
            
            let input_sample_rates = if is_input {
                self.supported_sample_rates(
                    idx,
                    info.max_input_channels,
                    info.default_low_input_latency,
                    info.default_sample_rate,
                    true,
                )
            } else {
                Vec::new()
            };

            let output_sample_rates = if is_output {
                self.supported_sample_rates(
                    idx,
                    info.max_output_channels,
                    info.default_low_output_latency,
                    info.default_sample_rate,
                    false,
                )
            } else {
                Vec::new()
            };

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
            
            devices.push((
                name.to_string(),
                device_type.to_string(),
                is_input,
                is_output,
                input_sample_rates,
                output_sample_rates,
            ));
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
    fn read_input(&mut self, py: Python, name: String, max_samples: usize) -> PyResult<Vec<u8>> {
        let stream = self.streams.get(&name)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Stream '{}' not found", name)
            ))?;
        
        let py_obj = stream.read_input(py, max_samples)?;
        
        // Convert PyBytes back to Vec<u8>
        let bytes = py_obj.downcast::<PyBytes>(py)?;
        Ok(bytes.as_bytes().to_vec())
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
    
    /// Interrupt output for a stream and return the position at which it was interrupted
    fn interrupt_output(&self, name: String) -> PyResult<f64> {
        let stream = self.streams.get(&name)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Stream '{}' not found", name)
            ))?;
        let interrupt_position = stream.interrupt_output();
        Ok(interrupt_position)
    }
    
    /// Clear the input buffer of a stream
    fn clear_input_buffer(&self, name: String) -> PyResult<()> {
        let stream = self.streams.get(&name)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Stream '{}' not found", name)
            ))?;
        stream.clear_input_buffer();
        Ok(())
    }

    /// Get last observed absolute input peak for diagnostics
    fn get_input_peak(&self, name: String) -> PyResult<f32> {
        let stream = self.streams.get(&name)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Stream '{}' not found", name)
            ))?;
        Ok(stream.get_last_input_peak())
    }
    
    /// Set input callback for a stream
    fn set_input_callback(&mut self, name: String, callback: PyObject) -> PyResult<()> {
        let stream = self.streams.get_mut(&name)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Stream '{}' not found", name)
            ))?;
        stream.set_input_callback(callback)
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

fn suppress_portaudio_errors<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    #[cfg(unix)]
    {
        use std::fs::OpenOptions;
        use std::os::unix::io::AsRawFd;
        use std::panic::{self, AssertUnwindSafe};

        static STDERR_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
        let lock = STDERR_GUARD.get_or_init(|| Mutex::new(())).lock().unwrap();

        let stderr_fd = libc::STDERR_FILENO;
        let backup_fd = unsafe { libc::dup(stderr_fd) };
        if backup_fd < 0 {
            drop(lock);
            return f();
        }

        let null_file = match OpenOptions::new().write(true).open("/dev/null") {
            Ok(file) => file,
            Err(_) => {
                unsafe {
                    libc::close(backup_fd);
                }
                drop(lock);
                return f();
            }
        };

        if unsafe { libc::dup2(null_file.as_raw_fd(), stderr_fd) } < 0 {
            unsafe {
                libc::close(backup_fd);
            }
            drop(lock);
            return f();
        }

        let result = panic::catch_unwind(AssertUnwindSafe(f));

        unsafe {
            libc::dup2(backup_fd, stderr_fd);
            libc::close(backup_fd);
        }
        drop(lock);

        match result {
            Ok(value) => value,
            Err(err) => panic::resume_unwind(err),
        }
    }

    #[cfg(not(unix))]
    {
        f()
    }
}
