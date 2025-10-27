# Audio AEC Library

High-performance audio I/O library for Python with integrated Acoustic Echo Cancellation (AEC), designed for real-time TTS/STT pipelines.

## Features

- **Precise Timing**: Know exactly how much audio has been played (sample-accurate)
- **Integrated AEC**: Built-in acoustic echo cancellation using SpeexDSP
- **Interruption Support**: Cancel output immediately without buffering delays
- **Low Latency**: Minimal buffering for real-time applications
- **Duplex Streaming**: Simultaneous input/output with synchronized timing
- **Cross-platform**: Works on Windows, macOS, and Linux

## Why This Library?

PyAudio and similar libraries have several limitations for modern voice AI applications:

1. **Opaque Buffering**: You can't know exactly how much audio has been played
2. **No Interruption**: Difficult to cancel queued audio immediately
3. **Poor AEC Support**: Hard to get precise timing needed for echo cancellation
4. **High Latency**: Large buffers required for smooth playback

This library solves these problems with a Rust core for performance and precise control.

## Architecture

```
┌─────────────────────────────────────────┐
│           Python API Layer              │
│  (Simple, Pythonic interface)           │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│         PyO3 Bindings Layer             │
│  (Automatic Python ↔ Rust conversion)   │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│          Rust Core Library              │
│  ┌────────────────────────────────┐    │
│  │   Timestamped Ring Buffers     │    │
│  │   (Sample-accurate position)   │    │
│  └────────────────────────────────┘    │
│  ┌────────────────────────────────┐    │
│  │   CPAL Audio I/O               │    │
│  │   (Cross-platform audio)       │    │
│  └────────────────────────────────┘    │
│  ┌────────────────────────────────┐    │
│  │   SpeexDSP Echo Canceller      │    │
│  │   (Acoustic echo removal)      │    │
│  └────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

## Building

### Prerequisites

1. **Rust toolchain**: Install from https://rustup.rs/
2. **Python 3.8+**: With pip installed
3. **Maturin**: Install with `pip install maturin`

### Build Steps

```bash
# Clone the repository
cd rust-audio-aec

# Build and install the Python module
maturin develop --release

# Or build a wheel for distribution
maturin build --release
```

### Platform-specific Notes

**macOS**: Requires CoreAudio (included with macOS)
**Linux**: Requires ALSA development files (`libasound2-dev` on Ubuntu/Debian). If on nix, see [nix instructions](README_NIX.md).
**Windows**: Requires WASAPI (included with Windows)

## Usage

### Basic Example

```python
from audio_aec import create_duplex_stream
import numpy as np

# Create a duplex stream with AEC
stream = create_duplex_stream(sample_rate=16000, enable_aec=True)
stream.start()

# Write audio for playback (TTS)
audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
stream.write(audio_data.astype(np.float32))

# Read captured audio (STT)
captured = stream.read(16000)  # Read 1 second

# Get exact playback position
position = stream.get_playback_position()
print(f"Played {position:.3f} seconds")

# Interrupt playback immediately
stream.interrupt()

stream.stop()
```

### TTS Pipeline Example

```python
from audio_aec import AudioStream, TtsStreamPlayer

# Setup
stream = AudioStream(sample_rate=16000, enable_aec=True)
tts_player = TtsStreamPlayer(stream)
stream.start()

# Stream TTS chunks as they're generated
for chunk in tts_generator.generate(text):
    tts_player.play_chunk(chunk)
    
    # Check if user interrupted
    if user_interrupted:
        tts_player.interrupt()
        break

# Get exact timing
print(f"Played: {stream.get_playback_position():.3f}s")
print(f"Buffered: {stream.get_buffered_duration():.3f}s")
```

### STT with Echo Cancellation

```python
# Setup with AEC enabled
stream = AudioStream(sample_rate=16000, enable_aec=True)

# Set callback for processed audio
def process_audio(audio_data):
    # Echo-cancelled audio ready for STT
    transcription = stt_model.transcribe(audio_data)
    print(f"User said: {transcription}")

stream.set_input_callback(process_audio)
stream.start()

# Play TTS while listening - echo will be removed!
stream.write(tts_audio)
```

## Performance

- **Latency**: < 10ms round-trip on most systems
- **CPU Usage**: < 5% for duplex streaming with AEC
- **Memory**: ~10MB for typical configuration
- **Sample Accuracy**: Exact sample position tracking

## Comparison with PyAudio

| Feature | PyAudio | This Library |
|---------|---------|--------------|
| Exact playback position | ❌ | ✅ |
| Interrupt output | ❌ | ✅ |
| Integrated AEC | ❌ | ✅ |
| Sample-accurate timing | ❌ | ✅ |
| Zero-copy buffers | ❌ | ✅ |
| Modern async API | ❌ | ✅ |

## Technical Details

### Buffer Management

The library uses lock-free ring buffers with atomic counters for sample-accurate position tracking. This allows:
- Precise timing without locks
- Minimal latency
- Thread-safe operation

### Echo Cancellation

Uses SpeexDSP's echo canceller with:
- Adaptive filter (2048-4096 taps)
- Automatic delay estimation
- Nonlinear processing for residual echo

### Cross-platform Audio

Built on CPAL (Cross-Platform Audio Library) for:
- Native performance on each platform
- Consistent API across OS
- Low-level control when needed

## Contributing

Contributions welcome! The codebase is structured as:

```
rust-audio-aec/
├── src/
│   ├── lib.rs           # Python module definition
│   ├── audio_engine.rs  # Main engine
│   ├── stream.rs        # Duplex stream implementation  
│   ├── buffer.rs        # Timestamped ring buffers
│   └── aec.rs          # Echo cancellation
├── python/
│   └── audio_aec/      # Python wrapper
└── examples/           # Usage examples
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- CPAL for cross-platform audio
- SpeexDSP for echo cancellation
- PyO3 for Python bindings
