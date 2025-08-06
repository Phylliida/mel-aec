use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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
    clear_requested: Arc<AtomicUsize>,
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
        let clear_requested = Arc::new(AtomicUsize::new(0));
        let start_time = Instant::now();
        
        let writer = TimestampedBuffer {
            producer: Some(producer),
            consumer: None,
            samples_written: samples_written.clone(),
            samples_read: samples_read.clone(),
            clear_requested: clear_requested.clone(),
            start_time,
            sample_rate,
        };
        
        let reader = TimestampedBuffer {
            producer: None,
            consumer: Some(consumer),
            samples_written: samples_written.clone(),
            samples_read: samples_read.clone(),
            clear_requested: clear_requested.clone(),
            start_time,
            sample_rate,
        };
        
        (writer, reader)
    }
    
    /// Write samples to the buffer
    pub fn write(&mut self, samples: &[f32]) -> usize {
        // Check if clear was requested
        if self.clear_requested.swap(0, Ordering::AcqRel) > 0 {
            // Clear the ring buffer by discarding all pending data
            if let Some(producer) = &mut self.producer {
                // Skip all pending data by advancing the producer position
                // This effectively discards unread data
                let current_read = self.samples_read.load(Ordering::Acquire);
                self.samples_written.store(current_read, Ordering::Release);
                
                // Note: We can't directly clear the producer in ringbuf 0.4
                // The consumer will handle skipping data on its side
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
        // Check if clear was requested - consumer should also respect it
        if self.clear_requested.load(Ordering::Acquire) > 0 {
            // Clear any pending data in consumer's view
            if let Some(consumer) = &mut self.consumer {
                // Get the number of available samples
                let available = consumer.occupied_len();
                // Skip all available samples
                consumer.skip(available);
                // Update the read counter to match what we've discarded
                self.samples_read.fetch_add(available as u64, Ordering::Release);
            }
            // Return 0 samples (silence) when interrupted
            return 0;
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
        // Ensure we don't get negative values that wrap around
        if written >= read {
            (written - read) as usize
        } else {
            0
        }
    }
    
    /// Request to clear/interrupt the buffer
    pub fn request_clear(&self) {
        self.clear_requested.store(1, Ordering::Release);
    }
    
    /// Get precise timestamp for AEC alignment
    pub fn get_timestamp_us(&self) -> u64 {
        self.start_time.elapsed().as_micros() as u64
    }
}