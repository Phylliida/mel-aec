"""
High-performance audio I/O library with integrated acoustic echo cancellation.

This library provides:
- Precise timing information for audio playback/capture
- Integrated acoustic echo cancellation (AEC)
- Support for interrupting audio output
- Low-latency duplex streaming
"""

# Import from the compiled Rust extension module
# The .so file is in the same directory as this __init__.py
from . import audio_aec
AudioEngine = audio_aec.AudioEngine
DuplexStream = audio_aec.DuplexStream  
StreamConfig = audio_aec.StreamConfig
import numpy as np
import struct
from typing import Optional, Callable, Tuple, List
import threading
import queue

__version__ = "0.1.0"

class AudioStream:
    """High-level Python interface for duplex audio streaming with AEC."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        buffer_size: int = 256,
        enable_aec: bool = True,
        aec_filter_length: int = 2048,
        input_device: str = None,
        output_device: str = None,
    ):
        """
        Initialize audio stream.
        
        Args:
            sample_rate: Sample rate in Hz (default 16000)
            channels: Number of channels (default 1 for mono)
            buffer_size: Buffer size in samples (default 256)
            enable_aec: Enable acoustic echo cancellation (default True)
            aec_filter_length: AEC filter length in samples (default 2048)
            input_device: Name of the input device (optional)
            output_device: Name of the output device (optional)
        """
        self.engine = AudioEngine()
        
        # Configure stream
        config = StreamConfig()
        config.sample_rate = sample_rate
        config.channels = channels
        config.buffer_size = buffer_size
        config.enable_aec = enable_aec
        config.aec_filter_length = aec_filter_length
        config.input_device = input_device
        config.output_device = output_device
        
        self.config = config
        self.sample_rate = sample_rate
        self.channels = channels
        
        # Create stream
        self.engine.create_stream("main", config)
        self.stream_name = "main"
        
        # Callback management
        self._input_callback = None
        self._input_queue = queue.Queue()
        self._running = False
        
    def start(self):
        """Start the audio stream."""
        self.engine.start_stream(self.stream_name)
        self._running = True
        
    def stop(self):
        """Stop the audio stream."""
        self._running = False
        self.engine.stop_stream(self.stream_name)
        
    def write(self, audio_data: np.ndarray) -> int:
        """
        Write audio data for playback.
        
        Args:
            audio_data: Audio samples as numpy array (float32, -1.0 to 1.0)
            
        Returns:
            Number of samples written
        """
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Ensure data is in range [-1, 1]
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Convert to bytes
        audio_bytes = audio_data.tobytes()
        
        return self.engine.write_output(self.stream_name, audio_bytes)
    
    def read(self, num_samples: int) -> np.ndarray:
        """
        Read captured audio data.
        
        Args:
            num_samples: Number of samples to read
            
        Returns:
            Audio samples as numpy array (float32)
        """
        audio_bytes = self.engine.read_input(self.stream_name, num_samples)
        
        # Convert bytes to numpy array
        if audio_bytes:
            return np.frombuffer(audio_bytes, dtype=np.float32)
        else:
            return np.array([], dtype=np.float32)
    
    def get_playback_position(self) -> float:
        """
        Get exact playback position in seconds.
        
        Returns:
            Current playback position in seconds
        """
        return self.engine.get_playback_position(self.stream_name)
    
    def get_buffered_duration(self) -> float:
        """
        Get duration of audio currently buffered for output.
        
        Returns:
            Buffered duration in seconds
        """
        buffered_samples = self.engine.get_output_buffer_size(self.stream_name)
        return buffered_samples / self.sample_rate
    
    def interrupt(self):
        """Interrupt/clear the output buffer immediately."""
        self.engine.interrupt_output(self.stream_name)
    
    def reset_aec(self):
        """Reset the acoustic echo canceller state."""
        if self.config.enable_aec:
            # AEC reset not yet implemented at engine level
            pass
    
    def set_input_callback(self, callback: Callable[[np.ndarray], None]):
        """
        Set a callback for processed input audio.
        
        Args:
            callback: Function that receives audio data as numpy array
        """
        def wrapper(audio_bytes):
            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
            callback(audio_data)
        
        self.engine.set_input_callback(self.stream_name, wrapper)
        self._input_callback = callback
    
    def list_devices(self) -> List[Tuple[str, str, bool, bool, List[float], List[float]]]:
        """
        List available audio devices.
        
        Returns:
            List of tuples (name, type, is_input, is_output, input_sample_rates, output_sample_rates)
        """
        return self.engine.list_devices()
    
    def get_supported_configs(self) -> List[Tuple[int, int]]:
        """
        Get supported sample rates and channel counts.
        
        Returns:
            List of tuples (sample_rate, channels)
        """
        return self.engine.get_supported_configs()


class TtsStreamPlayer:
    """
    Specialized player for TTS streaming with interruption support.
    """
    
    def __init__(self, stream: AudioStream):
        self.stream = stream
        self._play_queue = queue.Queue()
        self._playing = False
        self._play_thread = None
        
    def play_chunk(self, audio_chunk: np.ndarray):
        """Queue audio chunk for playback."""
        self._play_queue.put(audio_chunk)
        
        if not self._playing:
            self._start_playback()
    
    def interrupt(self):
        """Interrupt current playback and clear queue."""
        # Clear the queue
        while not self._play_queue.empty():
            try:
                self._play_queue.get_nowait()
            except queue.Empty:
                break
        
        # Interrupt stream output
        self.stream.interrupt()
    
    def _start_playback(self):
        """Start background playback thread."""
        if self._playing:
            return
        
        self._playing = True
        self._play_thread = threading.Thread(target=self._playback_worker)
        self._play_thread.daemon = True
        self._play_thread.start()
    
    def _playback_worker(self):
        """Worker thread for audio playback."""
        while self._playing:
            try:
                chunk = self._play_queue.get(timeout=0.1)
                self.stream.write(chunk)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Playback error: {e}")
                
    def stop(self):
        """Stop the playback thread."""
        self._playing = False
        if self._play_thread:
            self._play_thread.join(timeout=1.0)


# Convenience functions
def create_duplex_stream(
    sample_rate: int = 16000,
    enable_aec: bool = True,
    input_device: str = None,
    output_device: str = None
) -> AudioStream:
    """
    Create a duplex audio stream with sensible defaults.
    
    Args:
        sample_rate: Sample rate in Hz
        enable_aec: Enable acoustic echo cancellation
        
    Returns:
        Configured AudioStream instance
    """
    return AudioStream(
        sample_rate=sample_rate,
        channels=1,
        buffer_size=256,
        enable_aec=enable_aec,
        aec_filter_length=2048,
        input_device=input_device,
        output_device=output_device
    )


__all__ = [
    'AudioEngine',
    'AudioStream', 
    'TtsStreamPlayer',
    'StreamConfig',
    'DuplexStream',
    'create_duplex_stream',
]
