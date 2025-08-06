use pyo3::prelude::*;
use pyo3::types::PyBytes;
use portaudio as pa;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::buffer::TimestampedBuffer;
use crate::aec::{AecProcessor, AecConfig};

/// Configuration for audio streams
#[pyclass]
#[derive(Clone)]
pub struct StreamConfig {
    #[pyo3(get, set)]
    pub sample_rate: u32,
    #[pyo3(get, set)]
    pub channels: u16,
    #[pyo3(get, set)]
    pub buffer_size: usize,
    #[pyo3(get, set)]
    pub enable_aec: bool,
    #[pyo3(get, set)]
    pub aec_filter_length: usize,
}

#[pymethods]
impl StreamConfig {
    #[new]
    fn new() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            buffer_size: 1024,  // Increased from 256 to reduce crackling
            enable_aec: true,
            aec_filter_length: 2048,
        }
    }
}

/// Duplex audio stream with integrated AEC using PortAudio
#[pyclass]
pub struct DuplexStream {
    // PortAudio context
    pa: Arc<Mutex<pa::PortAudio>>,
    
    // Stream handles
    output_stream: Option<Arc<Mutex<pa::Stream<pa::NonBlocking, pa::Output<f32>>>>>,
    input_stream: Option<Arc<Mutex<pa::Stream<pa::NonBlocking, pa::Input<f32>>>>>,
    
    // Buffers with precise timing
    output_buffer_writer: Arc<Mutex<TimestampedBuffer>>,
    output_buffer_reader: Arc<Mutex<TimestampedBuffer>>,
    input_buffer_writer: Arc<Mutex<TimestampedBuffer>>,
    input_buffer_reader: Arc<Mutex<TimestampedBuffer>>,
    
    // AEC processor
    aec: Option<Arc<AecProcessor>>,
    
    // Stream control
    is_running: Arc<AtomicBool>,
    config: StreamConfig,
    
    // Callback for processed input
    input_callback: Arc<Mutex<Option<PyObject>>>,
}

#[pymethods]
impl DuplexStream {
    #[new]
    pub fn new(config: StreamConfig) -> PyResult<Self> {
        // Initialize PortAudio
        let pa = pa::PortAudio::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to initialize PortAudio: {}", e)
            )
        })?;
        
        // Create timestamped buffers - larger size for smoother playback
        let buffer_capacity = config.sample_rate as usize * 4; // 4 seconds (increased from 2)
        let (output_writer, output_reader) = TimestampedBuffer::new(
            buffer_capacity,
            config.sample_rate,
        );
        let (input_writer, input_reader) = TimestampedBuffer::new(
            buffer_capacity,
            config.sample_rate,
        );
        
        // Create AEC processor if enabled
        let aec = if config.enable_aec {
            let aec_config = AecConfig {
                filter_length: config.aec_filter_length,
                frame_size: config.buffer_size,
                sample_rate: config.sample_rate,
            };
            Some(Arc::new(AecProcessor::new(aec_config).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to create AEC: {}", e)
                )
            })?))
        } else {
            None
        };
        
        Ok(Self {
            pa: Arc::new(Mutex::new(pa)),
            output_stream: None,
            input_stream: None,
            output_buffer_writer: Arc::new(Mutex::new(output_writer)),
            output_buffer_reader: Arc::new(Mutex::new(output_reader)),
            input_buffer_writer: Arc::new(Mutex::new(input_writer)),
            input_buffer_reader: Arc::new(Mutex::new(input_reader)),
            aec,
            is_running: Arc::new(AtomicBool::new(false)),
            config,
            input_callback: Arc::new(Mutex::new(None)),
        })
    }
    
    /// Start the duplex audio stream with real audio I/O
    pub fn start(&mut self) -> PyResult<()> {
        if self.is_running.load(Ordering::Acquire) {
            return Ok(());
        }
        
        let pa = self.pa.lock().unwrap();
        
        // Get default devices
        let output_device = pa.default_output_device().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("No output device available: {}", e)
            )
        })?;
        
        let input_device = pa.default_input_device().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("No input device available: {}", e)
            )
        })?;
        
        // Get device info for latency settings
        let output_info = pa.device_info(output_device).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to get output device info: {}", e)
            )
        })?;
        
        let input_info = pa.device_info(input_device).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to get input device info: {}", e)
            )
        })?;
        
        // Setup output stream parameters with better latency
        let output_params = pa::StreamParameters::<f32>::new(
            output_device,
            self.config.channels as i32,
            true,  // interleaved
            // Use suggested high latency for stable playback
            output_info.default_high_output_latency,
        );
        
        // Setup input stream parameters with better latency
        let input_params = pa::StreamParameters::<f32>::new(
            input_device,
            self.config.channels as i32,
            true,  // interleaved
            // Use suggested high latency for stable capture
            input_info.default_high_input_latency,
        );
        
        // Create output stream with proper settings
        let output_reader = self.output_buffer_reader.clone();
        let aec_playback = self.aec.clone();
        
        let output_callback = move |pa::OutputStreamCallbackArgs { buffer, .. }| {
            let mut reader = output_reader.lock().unwrap();
            let read = reader.read(buffer);
            
            // Feed to AEC if enabled
            if let Some(aec) = &aec_playback {
                if read > 0 {
                    aec.feed_playback(&buffer[..read]);
                }
            }
            
            // Fill remainder with silence
            for sample in &mut buffer[read..] {
                *sample = 0.0;
            }
            
            pa::Continue
        };
        
        // Create output stream settings
        let output_settings = pa::OutputStreamSettings::new(
            output_params,
            self.config.sample_rate as f64,
            self.config.buffer_size as u32,
        );
        
        let mut output_stream = pa.open_non_blocking_stream(
            output_settings,
            output_callback,
        ).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to open output stream: {}", e)
            )
        })?;
        
        // Create input stream with proper settings
        let input_writer = self.input_buffer_writer.clone();
        let aec_capture = self.aec.clone();
        let input_callback_ref = self.input_callback.clone();
        
        let input_callback = move |pa::InputStreamCallbackArgs { buffer, .. }| {
            let processed_data = if let Some(aec) = &aec_capture {
                // Apply AEC
                let mut output = vec![0.0f32; buffer.len()];
                let _ = aec.process_capture(buffer, &mut output);
                output
            } else {
                buffer.to_vec()
            };
            
            // Write to buffer
            let mut writer = input_writer.lock().unwrap();
            writer.write(&processed_data);
            
            // Call Python callback if set
            if let Some(callback) = input_callback_ref.lock().unwrap().as_ref() {
                Python::with_gil(|py| {
                    let py_data = PyBytes::new(py, unsafe {
                        std::slice::from_raw_parts(
                            processed_data.as_ptr() as *const u8,
                            processed_data.len() * std::mem::size_of::<f32>(),
                        )
                    });
                    let _ = callback.call1(py, (py_data,));
                });
            }
            
            pa::Continue
        };
        
        // Create input stream settings
        let input_settings = pa::InputStreamSettings::new(
            input_params,
            self.config.sample_rate as f64,
            self.config.buffer_size as u32,
        );
        
        let mut input_stream = pa.open_non_blocking_stream(
            input_settings,
            input_callback,
        ).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to open input stream: {}", e)
            )
        })?;
        
        // Start streams
        output_stream.start().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to start output stream: {}", e)
            )
        })?;
        
        input_stream.start().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to start input stream: {}", e)
            )
        })?;
        
        self.output_stream = Some(Arc::new(Mutex::new(output_stream)));
        self.input_stream = Some(Arc::new(Mutex::new(input_stream)));
        self.is_running.store(true, Ordering::Release);
        
        Ok(())
    }
    
    /// Stop the audio streams
    pub fn stop(&mut self) -> PyResult<()> {
        self.is_running.store(false, Ordering::Release);
        
        // Stop output stream
        if let Some(stream) = &self.output_stream {
            let mut s = stream.lock().unwrap();
            let _ = s.stop();
            let _ = s.close();
        }
        
        // Stop input stream
        if let Some(stream) = &self.input_stream {
            let mut s = stream.lock().unwrap();
            let _ = s.stop();
            let _ = s.close();
        }
        
        self.output_stream = None;
        self.input_stream = None;
        
        Ok(())
    }
    
    /// Write audio data for playback
    pub fn write_output(&self, data: &[u8]) -> PyResult<usize> {
        // Convert bytes to f32
        let samples: Vec<f32> = data.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        let mut writer = self.output_buffer_writer.lock().unwrap();
        Ok(writer.write(&samples))
    }
    
    /// Read captured audio data
    pub fn read_input(&self, max_samples: usize) -> PyResult<Vec<u8>> {
        let mut samples = vec![0.0f32; max_samples];
        let mut reader = self.input_buffer_reader.lock().unwrap();
        let read = reader.read(&mut samples);
        
        // Convert to bytes
        let bytes: Vec<u8> = samples[..read]
            .iter()
            .flat_map(|&s| s.to_le_bytes())
            .collect();
        
        Ok(bytes)
    }
    
    /// Get the exact playback position in seconds
    pub fn get_playback_position(&self) -> f64 {
        let reader = self.output_buffer_reader.lock().unwrap();
        reader.position_seconds()
    }
    
    /// Get the number of samples currently buffered for output
    pub fn get_output_buffer_size(&self) -> usize {
        let reader = self.output_buffer_reader.lock().unwrap();
        reader.buffered_samples()
    }
    
    /// Interrupt/clear the output buffer
    pub fn interrupt_output(&self) {
        let reader = self.output_buffer_reader.lock().unwrap();
        reader.request_clear();
    }
    
    /// Set a Python callback for processed input audio
    fn set_input_callback(&mut self, callback: PyObject) {
        *self.input_callback.lock().unwrap() = Some(callback);
    }
    
    /// Reset the AEC state
    fn reset_aec(&self) -> PyResult<()> {
        if let Some(aec) = &self.aec {
            aec.reset();
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "AEC is not enabled"
            ))
        }
    }
}