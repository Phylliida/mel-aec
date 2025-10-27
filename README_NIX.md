# Installation

1. Add the following to your config (or add these packages to your existing list)

```nix
environment.systemPackages = with pkgs; [    
    # general deps
    git
    # rust deps
    cargo
    rustc
    # python deps
    zlib
    gcc
    # mel-aec deps
    speex
    cmake
    glibc
    libclang
    swig
]
```

2. Also, add this line to your config

```bash
# needed to run nix shell
nix.settings.experimental-features = "nix-command flakes";
```

3. Run this command to provide all the libraries needed in the shell

```bash
nix-shell -p libtool autoconf automake pkg-config python3 uv alsa-lib
```

It is necessary to do it this way (as opposed to adding to your config) because these are build dependencies.

In particular, alsa-lib is important here or the build will silently fail to include alsa as a dependency (even if it is in your system config), it will fallback to oss, and no devices will be displayed.

4. Initialize your python venv (if you don't already have one)

```bash
uv venv
source ./.venv/bin/activate
```

5. Install maturin

```bash
uv pip install maturin
```

6. Checkout the repo and navigate inside

```bash
git clone https://github.com/antra-tess/mel-aec.git
cd mel-aec
```

7. Compile (this also makes the compiled library available to your python venv)

```bash
export LIBCLANG_PATH=$(nix eval --raw --expr 'let pkgs = import <nixpkgs> {}; in "${pkgs.llvmPackages.libclang.lib}/lib"' --impure --extra-experimental-features "nix-command flakes") 
maturin develop --release
```

8. Fixup the python venv so binaries point to the correct places:

```bash
cd ..
nix shell github:GuillaumeDesforges/fix-python
fix-python --venv ./.venv
```

9. Now you can use audio_aec! Enjoy. Next time you start a terminal, just do

```bash
nix-shell -p python3 uv
source ./.venv/bin/activate
```

10. Test make sure things are working

```python
from audio_aec import create_duplex_stream
stream = create_duplex_stream()
print(stream.list_devices())
```

If you see some devices, everything is installed correctly.

Troubleshooting:

- Maturin segfaults:

Need to reinstall maturin, fix-python can break it:

```bash
uv pip uninstall maturin
rm -rf ~/.cache/
uv pip install maturin
```
