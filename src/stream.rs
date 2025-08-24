use pyo3::prelude::*;
use pyo3::types::PyBytes;
use portaudio as pa;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::atomic::AtomicU32;
use std::time::Instant;

use crate::buffer::TimestampedBuffer;
use crate::aec::{AecProcessor as WebRtcAecProcessor, AecConfig};
use crate::aec_trait::AecProcessor;
use crate::fdaf_aec_wrapper::FdafAecWrapper;
use crate::synchronized_aec::RingBufferSync;

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
    #[pyo3(get, set)]
    pub buffer_duration_seconds: u32,  // Maximum buffer duration in seconds
    #[pyo3(get, set)]
    pub aec_type: String,  // "webrtc" or "fdaf"
    #[pyo3(get, set)]
    pub input_device: Option<String>, // Preferred input device name
    #[pyo3(get, set)]
    pub output_device: Option<String>, // Preferred output device name
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
            buffer_duration_seconds: 30,  // Default to 30 seconds
            aec_type: "webrtc".to_string(),  // Default to WebRTC (production-grade)
            input_device: None,
            output_device: None,
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
    aec: Option<Arc<Mutex<dyn AecProcessor>>>,
    
    // Synchronization for AEC
    aec_sync: Option<Arc<Mutex<RingBufferSync>>>,

    // AEC frame alignment and accumulation
    aec_frame_size: usize,
    aec_capture_accum: Arc<Mutex<Vec<f32>>>,
    aec_render_accum: Arc<Mutex<Vec<f32>>>,
    aec_render_tmp: Arc<Mutex<Vec<f32>>>,

    // Background callback thread (decoupled from audio callback)
    callback_shutdown: Arc<AtomicBool>,
    callback_thread: Option<std::thread::JoinHandle<()>>,
    
    // Stream control
    is_running: Arc<AtomicBool>,
    config: StreamConfig,
    
    // Callback for processed input
    input_callback: Arc<Mutex<Option<PyObject>>>,
    
    // Track timing
    stream_start_time: Arc<Mutex<Option<Instant>>>,

    // Diagnostics
    last_input_peak: Arc<AtomicU32>, // scaled by 1e6
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
        
        // Create buffers - use configured duration
        let buffer_capacity = (config.sample_rate as usize * config.buffer_duration_seconds as usize) * config.channels as usize;
        
        let (output_writer, output_reader) = TimestampedBuffer::new(
            buffer_capacity,
            config.sample_rate,
        );
        
        let (input_writer, input_reader) = TimestampedBuffer::new(
            buffer_capacity,
            config.sample_rate,
        );
        
        // Create AEC processor if enabled (prefer WebRTC with fallback to FDAF)
        let aec: Option<Arc<Mutex<dyn AecProcessor>>> = if config.enable_aec {
            match config.aec_type.as_str() {
                "webrtc" => {
                    let aec_config = AecConfig {
                        sample_rate: config.sample_rate,
                        channels: config.channels as u32,
                        enable_aec: true,
                        enable_agc: false,
                        enable_ns: false,
                    };
                    match WebRtcAecProcessor::new(aec_config) {
                        Ok(processor) => Some(Arc::new(Mutex::new(processor)) as Arc<Mutex<dyn AecProcessor>>),
                        Err(err) => {
                            eprintln!("WebRTC AEC init failed ({}). Falling back to FDAF (mono only).", err);
                            match FdafAecWrapper::new(config.sample_rate, config.channels) {
                                Ok(proc2) => Some(Arc::new(Mutex::new(proc2)) as Arc<Mutex<dyn AecProcessor>>),
                                Err(e2) => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                                    format!("Failed to create AEC (WebRTC + fallback FDAF): {}; {}", err, e2)
                                )),
                            }
                        }
                    }
                }
                "fdaf" => {
                    let processor = FdafAecWrapper::new(config.sample_rate, config.channels).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                            format!("Failed to create FDAF AEC: {}", e)
                        )
                    })?;
                    Some(Arc::new(Mutex::new(processor)))
                }
                other => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Unknown AEC type: {}. Use 'webrtc' or 'fdaf'", other)
                    ));
                }
            }
        } else {
            None
        };
        
        // Create synchronization buffer only for FDAF (WebRTC manages delay internally)
        let aec_sync = if config.enable_aec && config.aec_type.as_str() == "fdaf" {
            Some(Arc::new(Mutex::new(RingBufferSync::new(
                config.sample_rate,
                200,  // 200ms max delay buffer
                50,   // 50ms expected delay
            ))))
        } else {
            None
        };

        // Frame size for 10ms (used by WebRTC) and as minimum alignment unit
        let aec_frame_size = (config.sample_rate / 100) as usize;
        
        Ok(Self {
            pa: Arc::new(Mutex::new(pa)),
            duplex_stream: None,
            output_buffer_writer: Arc::new(Mutex::new(output_writer)),
            output_buffer_reader: Arc::new(Mutex::new(output_reader)),
            input_buffer_writer: Arc::new(Mutex::new(input_writer)),
            input_buffer_reader: Arc::new(Mutex::new(input_reader)),
            aec,
            aec_sync,
            aec_frame_size,
            aec_capture_accum: Arc::new(Mutex::new(Vec::with_capacity(aec_frame_size * 4))),
            aec_render_accum: Arc::new(Mutex::new(Vec::with_capacity(aec_frame_size * 4))),
            aec_render_tmp: Arc::new(Mutex::new(Vec::new())),
            callback_shutdown: Arc::new(AtomicBool::new(false)),
            callback_thread: None,
            is_running: Arc::new(AtomicBool::new(false)),
            config,
            input_callback: Arc::new(Mutex::new(None)),
            stream_start_time: Arc::new(Mutex::new(None)),
            last_input_peak: Arc::new(AtomicU32::new(0)),
        })
    }
    
    /// Start the audio stream
    pub fn start(&mut self) -> PyResult<()> {
        if self.is_running.load(Ordering::SeqCst) {
            return Ok(());
        }
        
        let pa = self.pa.lock().unwrap();
        
        // Choose output device: preferred by name or default
        let output_device = if let Some(name) = &self.config.output_device {
            let mut found: Option<pa::DeviceIndex> = None;
            for dev in pa.devices().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to enumerate devices: {}", e)))? {
                let (idx, info) = dev.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to get device info: {}", e)))?;
                if info.max_output_channels > 0 && info.name == *name {
                    found = Some(idx);
                    break;
                }
            }
            found.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Output device not found: {}", name)))?
        } else {
            pa.default_output_device().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to get default output device: {}", e)
            )
        })?
        };
        
        // Choose input device: preferred by name or default
        let input_device = if let Some(name) = &self.config.input_device {
            let mut found: Option<pa::DeviceIndex> = None;
            for dev in pa.devices().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to enumerate devices: {}", e)))? {
                let (idx, info) = dev.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to get device info: {}", e)))?;
                if info.max_input_channels > 0 && info.name == *name {
                    found = Some(idx);
                    break;
                }
            }
            found.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Input device not found: {}", name)))?
        } else {
            pa.default_input_device().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to get default input device: {}", e)
            )
        })?
        };
        
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
        
        // eprintln!("DEBUG: Using input device: {} (index={:?})", input_info.name, input_device);
        // eprintln!("DEBUG: Input channels: max={}, using={}", input_info.max_input_channels, self.config.channels);
        // eprintln!("DEBUG: Sample rate: {}", self.config.sample_rate);
        // eprintln!("DEBUG: Default input latency: low={:?}, high={:?}", 
        //     input_info.default_low_input_latency, input_info.default_high_input_latency);
        
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
        let aec_sync = self.aec_sync.clone();
        // Decouple Python callback from audio callback
        let stream_start_time = self.stream_start_time.clone();
        let aec_frame_size = self.aec_frame_size;
        let aec_capture_accum = self.aec_capture_accum.clone();
        let aec_render_accum = self.aec_render_accum.clone();
        let aec_render_tmp = self.aec_render_tmp.clone();
        let last_input_peak = self.last_input_peak.clone();

        
        // CRITICAL: Single duplex callback for synchronized I/O
        let duplex_callback = move |args: pa::DuplexStreamCallbackArgs<'_, f32, f32>| {
            let in_buffer = args.in_buffer;
            let out_buffer = args.out_buffer;
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
            
            // Track input peak for diagnostics
            {
                let mut peak = 0.0f32;
                let mut sum = 0.0f32;
                let mut non_zero_count = 0usize;
                for &s in in_buffer.iter() {
                    let a = s.abs();
                    if a > peak { peak = a; }
                    sum += a;
                    if a > 0.0001 { non_zero_count += 1; }
                }
                let scaled = (peak * 1_000_000.0) as u32;
                last_input_peak.store(scaled, Ordering::Relaxed);
                
                // Debug logging - only print occasionally to avoid spam
                static mut DEBUG_COUNTER: u32 = 0;
                unsafe {
                    DEBUG_COUNTER += 1;
                    if DEBUG_COUNTER % 100 == 0 {  // Print every ~1 second at 48kHz/480 samples
                        // eprintln!("DEBUG: Input buffer len={}, peak={:.6}, avg={:.6}, non_zero={}/{}", 
                        //     in_buffer.len(), peak, 
                        //     if in_buffer.len() > 0 { sum / in_buffer.len() as f32 } else { 0.0 },
                        //     non_zero_count, in_buffer.len());
                    }
                }
            }

            // Process with AEC if enabled (aligned frames, no Python/GIL here)
            let processed_input = if let Some(aec) = &aec {
                // Maintain history of render for FDAF alignment
                if let Some(sync) = &aec_sync {
                    let mut sync_guard = sync.lock().unwrap();
                    sync_guard.push_render(out_buffer);
                }

                // Accumulate capture samples
                {
                    let mut cap_acc = aec_capture_accum.lock().unwrap();
                    cap_acc.extend_from_slice(in_buffer);
                }

                // Accumulate render samples depending on AEC type
                if aec_sync.is_some() {
                    // FDAF path: use delayed render
                    if let Some(sync) = &aec_sync {
                        let mut tmp = aec_render_tmp.lock().unwrap();
                        tmp.resize(in_buffer.len(), 0.0);
                        let sync_guard = sync.lock().unwrap();
                        sync_guard.fill_delayed_render(&mut tmp[..]);
                        let mut ren_acc = aec_render_accum.lock().unwrap();
                        ren_acc.extend_from_slice(&tmp[..]);
                    }
                } else {
                    // WebRTC path: feed actual render now
                    let mut ren_acc = aec_render_accum.lock().unwrap();
                    ren_acc.extend_from_slice(out_buffer);
                }

                // Process in aec_frame_size chunks
                let mut produced: Vec<f32> = Vec::with_capacity(in_buffer.len());
                let mut aec_guard = aec.lock().unwrap();
                loop {
                    let ready = { aec_capture_accum.lock().unwrap().len() };
                    if ready < aec_frame_size { break; }

                    // Extract one frame from accumulators
                    let mut cap_frame = vec![0.0f32; aec_frame_size];
                    let mut ren_frame = vec![0.0f32; aec_frame_size];
                    {
                        let mut cap_acc = aec_capture_accum.lock().unwrap();
                        cap_frame.copy_from_slice(&cap_acc[..aec_frame_size]);
                        cap_acc.drain(..aec_frame_size);
                    }
                    {
                        let mut ren_acc = aec_render_accum.lock().unwrap();
                        if ren_acc.len() >= aec_frame_size {
                            ren_frame.copy_from_slice(&ren_acc[..aec_frame_size]);
                            ren_acc.drain(..aec_frame_size);
                        } else {
                            for v in ren_frame.iter_mut() { *v = 0.0; }
                        }
                    }

                    if let Err(e) = aec_guard.process_render(&ren_frame[..]) {
                        eprintln!("AEC render error: {}", e);
                    }

                    let mut out_frame = vec![0.0f32; aec_frame_size];
                    match aec_guard.process_capture(&cap_frame[..], &mut out_frame[..]) {
                        Ok(_) => produced.extend_from_slice(&out_frame[..]),
                        Err(e) => { eprintln!("AEC capture error: {}", e); produced.extend_from_slice(&cap_frame[..]); }
                    }
                }

                produced
            } else {
                in_buffer.to_vec()
            };
            
            // Write processed input to buffer
            let mut input_writer = input_writer.lock().unwrap();
            input_writer.write(&processed_input);
            
            // Python callback handled by a background thread to avoid GIL jitter in RT path
            
            pa::Continue
        };
        
        // Let PortAudio/CoreAudio choose frames-per-buffer; we'll align to 10ms internally
        let callback_frames = 0u32; // pa::FRAMES_PER_BUFFER_UNSPECIFIED
        // Create duplex stream settings
        let duplex_settings = pa::DuplexStreamSettings::new(
            input_params,
            output_params,
            self.config.sample_rate as f64,
            callback_frames,
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
        // Release PortAudio mutex guard before mutating self further
        drop(pa);

        self.duplex_stream = Some(Arc::new(Mutex::new(duplex_stream)));
        self.is_running.store(true, Ordering::SeqCst);
        
        // Reset timing
        *self.stream_start_time.lock().unwrap() = Some(Instant::now());
        
        // Start background callback thread if set
        self.maybe_start_callback_thread();
        
        Ok(())
    }
    
    /// Stop the audio stream
    pub fn stop(&mut self) -> PyResult<()> {
        self.is_running.store(false, Ordering::SeqCst);
        
        // Stop background callback thread
        self.callback_shutdown.store(true, Ordering::SeqCst);
        if let Some(handle) = self.callback_thread.take() { let _ = handle.join(); }

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
    
    /// Clear the input buffer (useful for discarding pre-recorded audio)
    pub fn clear_input_buffer(&self) {
        let mut reader = self.input_buffer_reader.lock().unwrap();
        // Read and discard all available samples
        let available = reader.buffered_samples();
        if available > 0 {
            let mut discard = vec![0.0f32; available];
            reader.read(&mut discard);
        }
    }
    
    /// Interrupt/clear the output buffer
    pub fn interrupt_output(&self) -> f64 {
        let position = self.get_playback_position();
        // Coordinate an interrupt by signaling the writer side
        {
            let writer = self.output_buffer_writer.lock().unwrap();
            writer.request_clear();
        }
        
        position
    }
    
    /// Set a callback for processed input data
    pub fn set_input_callback(&mut self, callback: PyObject) -> PyResult<()> {
        *self.input_callback.lock().unwrap() = Some(callback);
        if self.is_running.load(Ordering::SeqCst) && self.callback_thread.is_none() {
            self.maybe_start_callback_thread();
        }
        Ok(())
    }
    
    /// Check if the stream is currently running
    pub fn is_active(&self) -> bool {
        self.is_running.load(Ordering::SeqCst)
    }

    fn maybe_start_callback_thread(&mut self) {
        if self.callback_thread.is_some() { return; }
        if self.input_callback.lock().unwrap().is_none() { return; }
        let reader = self.input_buffer_reader.clone();
        let cb_ref = self.input_callback.clone();
        let shutdown = self.callback_shutdown.clone();
        let chunk = self.aec_frame_size.max(1);
        self.callback_shutdown.store(false, Ordering::SeqCst);
        self.callback_thread = Some(std::thread::spawn(move || {
            use std::time::Duration;
            loop {
                if shutdown.load(Ordering::SeqCst) { break; }
                let mut data: Vec<f32> = Vec::with_capacity(chunk);
                let read = {
                    let mut r = reader.lock().unwrap();
                    let mut tmp = vec![0.0f32; chunk];
                    let n = r.read(&mut tmp);
                    if n > 0 { data.extend_from_slice(&tmp[..n]); }
                    n
                };
                if read > 0 {
                    if let Some(callback) = cb_ref.lock().unwrap().as_ref() {
                        Python::with_gil(|py| {
                            let py_data = PyBytes::new(py, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<f32>(),
                                )
                            });
                            let _ = callback.call1(py, (py_data,));
                        });
                    }
                } else {
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        }));
    }

    /// Get the last observed absolute input peak in the audio callback
    pub fn get_last_input_peak(&self) -> f32 {
        (self.last_input_peak.load(Ordering::Relaxed) as f32) / 1_000_000.0
    }
}
