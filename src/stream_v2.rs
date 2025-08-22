use pyo3::prelude::*;
use pyo3::types::PyBytes;
use portaudio as pa;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::buffer::TimestampedBuffer;
use crate::aec::{AecProcessor, AecConfig};

/// Configuration for audio streams
#[pyclass]
#[derive(Debug, Clone)]
pub struct StreamConfig {
    #[pyo3(get, set)]
    pub sample_rate: u32,
    #[pyo3(get, set)]
    pub channels: u16,
    #[pyo3(get, set)]
    pub buffer_size: u32,
    #[pyo3(get, set)]
    pub enable_aec: bool,
    #[pyo3(get, set)]
    pub aec_filter_length: u32,
}

#[pymethods]
impl StreamConfig {
    #[new]
    fn new() -> Self {
        Self::default()
    }
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,  // WebRTC standard
            channels: 1,
            buffer_size: 480,    // 10ms at 48kHz
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
    
    // Single duplex stream
    duplex_stream: Option<Arc<Mutex<pa::Stream<pa::NonBlocking, pa::Duplex<f32, f32>>>>>,
    
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
    
    // Track timing
    stream_start_time: Arc<Mutex<Option<Instant>>>,
}

#[pymethods]
impl DuplexStream {
    #[new]
    pub fn new(config: StreamConfig) -> PyResult<Self> {
        let pa = pa::PortAudio::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to initialize PortAudio: {}", e)
            )
        })?;
        
        // Create buffers
        let buffer_capacity = (config.sample_rate as usize * 2) * config.channels as usize;
        
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
                sample_rate: config.sample_rate,
                channels: config.channels as u32,
                enable_aec: true,
                enable_agc: false,
                enable_ns: false,
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
            duplex_stream: None,
            output_buffer_writer: Arc::new(Mutex::new(output_writer)),
            output_buffer_reader: Arc::new(Mutex::new(output_reader)),
            input_buffer_writer: Arc::new(Mutex::new(input_writer)),
            input_buffer_reader: Arc::new(Mutex::new(input_reader)),
            aec,
            is_running: Arc::new(AtomicBool::new(false)),
            config,
            input_callback: Arc::new(Mutex::new(None)),
            stream_start_time: Arc::new(Mutex::new(None)),
        })
    }
    
    /// Start the audio stream
    pub fn start(&mut self) -> PyResult<()> {
        if self.is_running.load(Ordering::SeqCst) {
            return Ok(());
        }
        
        let pa = self.pa.lock().unwrap();
        
        // Get default devices
        let output_device = pa.default_output_device().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to get default output device: {}", e)
            )
        })?;
        
        let input_device = pa.default_input_device().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to get default input device: {}", e)
            )
        })?;
        
        // Get device info
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
        
        // CRITICAL: Use LOW latency for precise timing
        let output_params = pa::StreamParameters::<f32>::new(
            output_device,
            self.config.channels as i32,
            true,  // interleaved
            output_info.default_low_output_latency,  // LOW latency!
        );
        
        let input_params = pa::StreamParameters::<f32>::new(
            input_device,
            self.config.channels as i32,
            true,  // interleaved
            input_info.default_low_input_latency,    // LOW latency!
        );
        
        // Clone references for the callback
        let output_reader = self.output_buffer_reader.clone();
        let input_writer = self.input_buffer_writer.clone();
        let aec = self.aec.clone();
        let input_callback_ref = self.input_callback.clone();
        let stream_start_time = self.stream_start_time.clone();
        
        // CRITICAL: Single duplex callback for synchronized I/O
        let duplex_callback = move |pa::DuplexStreamCallbackArgs { 
            in_buffer, 
            out_buffer, 
            frames,
            .. 
        }| {
            // Track stream start time
            let mut start_guard = stream_start_time.lock().unwrap();
            if start_guard.is_none() {
                *start_guard = Some(Instant::now());
            }
            drop(start_guard);
            
            // First, get output data for speakers
            let mut output_reader = output_reader.lock().unwrap();
            let output_samples = output_reader.read(out_buffer);
            
            // Fill any remaining buffer with silence
            for sample in &mut out_buffer[output_samples..] {
                *sample = 0.0;
            }
            
            // CRITICAL: Process render through AEC BEFORE capture
            if let Some(aec) = &aec {
                if output_samples > 0 {
                    let _ = aec.process_render(&out_buffer[..output_samples]);
                }
            }
            
            // Now process capture with AEC (echo should be removed)
            let processed_input = if let Some(aec) = &aec {
                let mut processed = vec![0.0f32; in_buffer.len()];
                let _ = aec.process_capture(in_buffer, &mut processed);
                processed
            } else {
                in_buffer.to_vec()
            };
            
            // Write processed input to buffer
            let mut input_writer = input_writer.lock().unwrap();
            input_writer.write(&processed_input);
            
            // Call Python callback if set
            if let Some(callback) = input_callback_ref.lock().unwrap().as_ref() {
                Python::with_gil(|py| {
                    let py_data = PyBytes::new(py, unsafe {
                        std::slice::from_raw_parts(
                            processed_input.as_ptr() as *const u8,
                            processed_input.len() * std::mem::size_of::<f32>(),
                        )
                    });
                    let _ = callback.call1(py, (py_data,));
                });
            }
            
            pa::Continue
        };
        
        // Create duplex stream settings
        let duplex_settings = pa::DuplexStreamSettings::new(
            input_params,
            output_params,
            self.config.sample_rate as f64,
            self.config.buffer_size as u32,
        );
        
        // Open duplex stream
        let mut duplex_stream = pa.open_non_blocking_stream(
            duplex_settings,
            duplex_callback,
        ).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to open duplex stream: {}", e)
            )
        })?;
        
        // Start the stream
        duplex_stream.start().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to start duplex stream: {}", e)
            )
        })?;
        
        self.duplex_stream = Some(Arc::new(Mutex::new(duplex_stream)));
        self.is_running.store(true, Ordering::SeqCst);
        
        // Reset timing
        *self.stream_start_time.lock().unwrap() = Some(Instant::now());
        
        Ok(())
    }
    
    /// Stop the audio stream
    pub fn stop(&mut self) -> PyResult<()> {
        self.is_running.store(false, Ordering::SeqCst);
        
        if let Some(stream) = &self.duplex_stream {
            let mut stream = stream.lock().unwrap();
            stream.stop().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to stop stream: {}", e)
                )
            })?;
        }
        
        self.duplex_stream = None;
        *self.stream_start_time.lock().unwrap() = None;
        
        Ok(())
    }
    
    /// Write audio data for playback
    pub fn write_output(&self, data: &[u8]) -> PyResult<usize> {
        // Convert bytes to f32
        let samples: Vec<f32> = data.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        let mut writer = self.output_buffer_writer.lock().unwrap();
        let written = writer.write(&samples);
        
        Ok(written)
    }
    
    /// Read captured audio data
    pub fn read_input(&self, py: Python, max_samples: usize) -> PyResult<PyObject> {
        let mut reader = self.input_buffer_reader.lock().unwrap();
        
        let mut buffer = vec![0.0f32; max_samples];
        let read = reader.read(&mut buffer);
        
        if read > 0 {
            buffer.truncate(read);
            let bytes: Vec<u8> = buffer.iter()
                .flat_map(|&sample| sample.to_le_bytes())
                .collect();
            
            Ok(PyBytes::new(py, &bytes).into())
        } else {
            Ok(PyBytes::new(py, &[]).into())
        }
    }
    
    /// Get the exact playback position in seconds
    pub fn get_playback_position(&self) -> f64 {
        let reader = self.output_buffer_reader.lock().unwrap();
        reader.position_seconds()
    }
    
    /// Get the exact capture position in seconds
    pub fn get_capture_position(&self) -> f64 {
        let writer = self.input_buffer_writer.lock().unwrap();
        writer.position_seconds()
    }
    
    /// Get stream time since start
    pub fn get_stream_time(&self) -> f64 {
        if let Some(start) = *self.stream_start_time.lock().unwrap() {
            start.elapsed().as_secs_f64()
        } else {
            0.0
        }
    }
    
    /// Get the current output buffer size
    pub fn get_output_buffer_size(&self) -> usize {
        let reader = self.output_buffer_reader.lock().unwrap();
        reader.buffered_samples()
    }
    
    /// Interrupt/clear the output buffer
    pub fn interrupt_output(&self) -> f64 {
        let position = self.get_playback_position();
        
        let mut reader = self.output_buffer_reader.lock().unwrap();
        reader.clear();
        
        position
    }
    
    /// Set a callback for processed input data
    pub fn set_input_callback(&mut self, callback: PyObject) -> PyResult<()> {
        *self.input_callback.lock().unwrap() = Some(callback);
        Ok(())
    }
    
    /// Check if the stream is currently running
    pub fn is_active(&self) -> bool {
        self.is_running.load(Ordering::SeqCst)
    }
}
