# Custom TTS with TorchDevice

This project demonstrates using Coqui TTS (Text-to-Speech) with TorchDevice to redirect CUDA calls to MPS on Apple Silicon hardware. It allows for generating speech with customizable voices using PyTorch-based models.

## Features

- Basic text-to-speech generation
- Voice customization and cloning
- Multilingual support
- Seamless integration with TorchDevice for hardware acceleration on Apple Silicon

## Requirements

- Python 3.8+
- PyTorch 2.0+
- TorchDevice library
- Coqui TTS

## Scripts

### 1. Basic TTS Test

`tts_with_torchdevice.py` - Tests basic TTS functionality with TorchDevice integration.

```bash
python tts_with_torchdevice.py
```

This script:
- Lists available TTS models
- Tests basic TTS with a simple English model
- Optionally tests multilingual TTS

### 2. Custom Voice TTS

`custom_voice_tts.py` - Demonstrates voice customization and cloning capabilities.

```bash
# List models that support voice cloning
python custom_voice_tts.py --list-models

# Generate speech with a custom voice
python custom_voice_tts.py --text "Your text here" --speaker path/to/speaker_sample.wav --output samples/output.wav

# Generate speech in a different language
python custom_voice_tts.py --text "Bonjour, comment ça va?" --speaker path/to/speaker_sample.wav --language fr --output samples/french_output.wav
```

## How It Works

1. TorchDevice is imported first to intercept CUDA calls
2. When TTS models attempt to use CUDA functions, TorchDevice redirects them to MPS
3. This allows PyTorch models designed for CUDA to run on Apple Silicon without code modifications

## Sample Usage

```python
import torch
import TorchDevice  # Import first to intercept CUDA calls
from TTS.api import TTS

# Initialize TTS with a model
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")

# Generate speech
tts.tts_to_file(text="Hello, this is a test.", file_path="output.wav")

# For voice cloning
tts = TTS("tts_models/multilingual/multi-dataset/your_tts")
tts.tts_to_file(
    text="This is voice cloning.",
    speaker_wav="path/to/speaker_sample.wav",
    language="en",
    file_path="cloned_voice.wav"
)
```

## Notes

- Some models may require downloading additional files on first use
- Voice cloning quality depends on the quality of the speaker sample
- Performance may vary based on the complexity of the model and the hardware
