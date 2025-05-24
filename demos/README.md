# TorchDevice Demos

This directory contains demo scripts for running language models and TTS models using TorchDevice. These scripts are designed to:
- Demonstrate device-agnostic inference (CPU, MPS, CUDA)
- Work with small Hugging Face models for quick testing
- Serve as proof-of-concept for integration with real-world LLM and TTS workflows

## Demo Scripts

- `demo_llasa3b_tts.py`: Runs a TTS model (LLASA3B or fallback) and logs device info, timing, and output. 
- `demo_transformers_small.py`: Runs a small masked language model (e.g., tiny-random-bert) and logs device info, timing, and output.

## Running the Demos

```bash
python demos/demo_llasa3b_tts.py
python demos/demo_transformers_small.py
```

Both scripts will automatically select the best available device (CUDA, MPS, or CPU) and use a small model by default.

### Using Different Models

You can specify a different model by setting an environment variable:

- For TTS demo:
  ```bash
  LLASA3B_MODEL="your/model-name" python demos/demo_llasa3b_tts.py
  ```
- For LLM demo:
  ```bash
  TRANSFORMERS_MODEL="your/model-name" python demos/demo_transformers_small.py
  ```

> **Note:** Use small models (e.g., `hf-internal-testing/tiny-random-bert`) for quick tests. Large models may require significant memory and download time.

## Hugging Face Hub Caching

The scripts use Hugging Face's `from_pretrained` method, which will cache models locally after the first download. You can control the cache location with the `HF_HOME` environment variable:

```bash
export HF_HOME=~/.cache/huggingface
```

## Customization & Hyperparameters

You can edit the scripts to change inference hyperparameters (e.g., `max_length`, `top_p`, `temperature`) or input text. See the script source for details.

## Next Steps
- Try running the demos on different devices.
- Experiment with different models and hyperparameters.
- Use these as a starting point for interactive or chat-based demos.

---

For more information, see the main project README or contact the maintainers. 