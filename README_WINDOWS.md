# Windows Installation Guide for audio-aec

## Prerequisites

1. **Python 3.8 or higher**
   - Download from https://python.org
   - Make sure to check "Add Python to PATH" during installation

2. **Rust toolchain**
   ```powershell
   # Download and run rustup-init.exe from https://rustup.rs
   # Or use winget:
   winget install Rustlang.Rustup
   ```

3. **Visual Studio Build Tools**
   - Download from https://visualstudio.microsoft.com/downloads/
   - Install "Desktop development with C++" workload
   - Or use the lighter Build Tools for Visual Studio

## Installation

1. **Clone the repository**
   ```powershell
   git clone <repository-url>
   cd rust-audio-aec
   ```

2. **Install maturin**
   ```powershell
   pip install maturin
   ```

3. **Build and install the module**
   ```powershell
   # For development (debug build, faster compilation)
   maturin develop

   # For production (optimized build)
   maturin develop --release
   ```

## Testing

1. **Test basic recording**
   ```powershell
   python examples/record_voice.py --output test.wav --duration 3
   ```

2. **Test AEC functionality**
   ```powershell
   python examples/test_aec_residual.py --file test.wav
   ```

## Windows-Specific Notes

### Audio Playback
The example scripts use `afplay` (macOS) for audio playback. On Windows, you can:
- Use Windows Media Player: `start test.wav`
- Use PowerShell: `[System.Media.SoundPlayer]::new('test.wav').PlaySync()`
- Install a Python library: `pip install playsound` then `playsound('test.wav')`

### Default Audio Devices
Windows will use the default audio input/output devices configured in Sound Settings.
To list available devices:
```python
from audio_aec import AudioEngine

engine = AudioEngine()
devices = engine.list_devices()
for name, dev_type, is_input, is_output in devices:
    print(f"{name}: {'input' if is_input else ''} {'output' if is_output else ''}")
```

### Potential Issues

1. **PortAudio initialization error**
   - Make sure Windows Audio service is running
   - Check that you have audio devices enabled

2. **Build errors with WebRTC**
   - Ensure Visual Studio Build Tools are installed
   - The `bundled` feature should handle compilation automatically

3. **Permission issues**
   - Windows may prompt for microphone access on first use
   - Run PowerShell/CMD as Administrator if needed

## Performance

For best performance on Windows:
- Use WASAPI host API (PortAudio default on Windows)
- Close other audio applications to avoid conflicts
- Use 48000 Hz sample rate (typical Windows default)

## Troubleshooting

If you encounter issues:
1. Check that all audio services are running:
   ```powershell
   Get-Service | Where {$_.Name -like "*audio*"}
   ```

2. Verify PortAudio can see your devices:
   ```python
   import portaudio
   pa = portaudio.PortAudio()
   for i in range(pa.device_count):
       info = pa.device_info(i)
       print(f"{i}: {info.name}")
   ```

3. Check Windows Event Viewer for audio-related errors

## Distribution

To create a wheel for distribution:
```powershell
maturin build --release
# Wheel will be in target/wheels/
```

The resulting wheel can be installed on any Windows machine with:
```powershell
pip install audio_aec-0.1.0-cp38-none-win_amd64.whl
```