brew install portaudio
brew install libtool
brew install automake
brew install pkg-config
brew install llvm
brew install cmake

brew install ffmpeg

uv pip install 'nemo_toolkit[asr]'
uv pip install nemo_toolkit --upgrade
uv pip install transformers --upgrade

python 3.13 with uv


uv venv --python 3.13 .venv

uv pip install maturin

uv pip install resampy
uv pip install torch
uv pip install librosa
uv pip install rapidfuzz
uv pip install matplotlib --upgrade
upgrade deepgram-sdk
