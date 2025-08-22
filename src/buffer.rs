use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use ringbuf::{HeapRb, HeapCons as HeapConsumer, HeapProd as HeapProducer};
use ringbuf::traits::{Split, Producer, Consumer, Observer};

/// A buffer that tracks precise playback position for audio samples
pub struct TimestampedBuffer {
    // Ring buffer for audio data
    producer: Option<HeapProducer<f32>>,
    consumer: Option<HeapConsumer<f32>>,
    
    // Atomic counters for position tracking
    samples_written: Arc<AtomicU64>,
    samples_read: Arc<AtomicU64>,
    
    // Interrupt tracking with generation counter to avoid race conditions
    interrupt_generation: Arc<AtomicU64>,
    last_handled_interrupt_write: Arc<AtomicU64>,
    last_handled_interrupt_read: Arc<AtomicU64>,
    
    // Track the position at which the interrupt occurred
    interrupt_position: Arc<AtomicU64>,
    
    start_time: Instant,
    
    // Sample rate for time calculations
    sample_rate: u32,
}

impl TimestampedBuffer {
    /// Create a new pair of writer/reader buffers
    pub fn new(capacity: usize, sample_rate: u32) -> (Self, Self) {
        let rb = HeapRb::<f32>::new(capacity);
        let (producer, consumer) = rb.split();
        
        let samples_written = Arc::new(AtomicU64::new(0));
        let samples_read = Arc::new(AtomicU64::new(0));
        let interrupt_generation = Arc::new(AtomicU64::new(0));
        let last_handled_interrupt_write = Arc::new(AtomicU64::new(0));
        let last_handled_interrupt_read = Arc::new(AtomicU64::new(0));
        let interrupt_position = Arc::new(AtomicU64::new(0));
        let start_time = Instant::now();
        
        let writer = TimestampedBuffer {
            producer: Some(producer),
            consumer: None,
            samples_written: samples_written.clone(),
            samples_read: samples_read.clone(),
            interrupt_generation: interrupt_generation.clone(),
            last_handled_interrupt_write: last_handled_interrupt_write.clone(),
            last_handled_interrupt_read: last_handled_interrupt_read.clone(),
            interrupt_position: interrupt_position.clone(),
            start_time,
            sample_rate,
        };
        
        let reader = TimestampedBuffer {
            producer: None,
            consumer: Some(consumer),
            samples_written: samples_written.clone(),
            samples_read: samples_read.clone(),
            interrupt_generation: interrupt_generation.clone(),
            last_handled_interrupt_write: last_handled_interrupt_write.clone(),
            last_handled_interrupt_read: last_handled_interrupt_read.clone(),
            interrupt_position: interrupt_position.clone(),
            start_time,
            sample_rate,
        };
        
        (writer, reader)
    }
    
    /// Write samples to the buffer
    pub fn write(&mut self, samples: &[f32]) -> usize {
        // Check if there's a new interrupt to handle
        let current_interrupt = self.interrupt_generation.load(Ordering::Acquire);
        let last_handled = self.last_handled_interrupt_write.load(Ordering::Acquire);
        
        if current_interrupt > last_handled {
            // Handle the interrupt
            if let Some(_producer) = &mut self.producer {
                // Clear any pending data in the producer
                // We can't directly clear the ringbuf, but we can stop writing new data
                // and update our position tracking
                
                // Save the current read position as the interrupt position
                let current_read = self.samples_read.load(Ordering::Acquire);
                self.interrupt_position.store(current_read, Ordering::Release);
                
                // Reset the write position to match the read position
                self.samples_written.store(current_read, Ordering::Release);
                
                // Mark this interrupt as handled by the writer
                self.last_handled_interrupt_write.store(current_interrupt, Ordering::Release);
            }
        }
        
        if let Some(producer) = &mut self.producer {
            let written = producer.push_slice(samples);
            self.samples_written.fetch_add(written as u64, Ordering::Release);
            written
        } else {
            0
        }
    }
    
    /// Read samples from the buffer
    pub fn read(&mut self, samples: &mut [f32]) -> usize {
        // Check if there's a new interrupt to handle
        let current_interrupt = self.interrupt_generation.load(Ordering::Acquire);
        let last_handled = self.last_handled_interrupt_read.load(Ordering::Acquire);
        
        if current_interrupt > last_handled {
            // Handle the interrupt
            if let Some(consumer) = &mut self.consumer {
                // Clear any pending data in consumer's view
                let available = consumer.occupied_len();
                if available > 0 {
                    // Skip all available samples
                    consumer.skip(available);
                    // Update the read counter
                    self.samples_read.fetch_add(available as u64, Ordering::Release);
                }
                
                // Mark this interrupt as handled by the reader
                self.last_handled_interrupt_read.store(current_interrupt, Ordering::Release);
                
                // Fill the output with silence for this callback
                for sample in samples.iter_mut() {
                    *sample = 0.0;
                }
                return samples.len(); // Return that we "read" silence
            }
        }
        
        if let Some(consumer) = &mut self.consumer {
            let read = consumer.pop_slice(samples);
            self.samples_read.fetch_add(read as u64, Ordering::Release);
            read
        } else {
            0
        }
    }
    
    /// Get the exact number of samples that have been played
    pub fn samples_played(&self) -> u64 {
        self.samples_read.load(Ordering::Acquire)
    }
    
    /// Get the playback position in seconds
    pub fn position_seconds(&self) -> f64 {
        self.samples_played() as f64 / self.sample_rate as f64
    }
    
    /// Get the number of samples currently buffered
    pub fn buffered_samples(&self) -> usize {
        let written = self.samples_written.load(Ordering::Acquire);
        let read = self.samples_read.load(Ordering::Acquire);
        
        // Check if we're in an interrupted state
        let current_interrupt = self.interrupt_generation.load(Ordering::Acquire);
        let write_handled = self.last_handled_interrupt_write.load(Ordering::Acquire);
        let read_handled = self.last_handled_interrupt_read.load(Ordering::Acquire);
        
        // If both sides have handled the interrupt, use normal calculation
        if current_interrupt == write_handled && current_interrupt == read_handled {
            if written >= read {
                (written - read) as usize
            } else {
                0
            }
        } else {
            // During interrupt processing, report 0 buffered to avoid confusion
            0
        }
    }
    
    /// Request to clear/interrupt the buffer
    pub fn request_clear(&self) {
        // Increment the generation counter to signal a new interrupt
        self.interrupt_generation.fetch_add(1, Ordering::Release);
    }
    
    /// Get the position at which the last interrupt occurred
    pub fn get_interrupt_position(&self) -> f64 {
        let pos_samples = self.interrupt_position.load(Ordering::Acquire);
        pos_samples as f64 / self.sample_rate as f64
    }
    
    /// Check if an interrupt is currently being processed
    pub fn is_interrupt_pending(&self) -> bool {
        let current = self.interrupt_generation.load(Ordering::Acquire);
        let write_handled = self.last_handled_interrupt_write.load(Ordering::Acquire);
        let read_handled = self.last_handled_interrupt_read.load(Ordering::Acquire);
        
        current > write_handled || current > read_handled
    }
    
    /// Get precise timestamp for AEC alignment
    pub fn get_timestamp_us(&self) -> u64 {
        self.start_time.elapsed().as_micros() as u64
    }
}