# Alice GPT

A minimal, character-level (or BPE) GPT implementation trained on *Alice in Wonderland*, inspired by nanoGPT. 

This project trains a small transformer-based language model (~2M parameters) from scratch to generate text in the style of Alice in Wonderland. It is designed to be lightweight enough to train quickly on a single GPU (e.g., RTX 3050) or even a CPU.

## Project Structure

- `prepare.py`: Tokenizes the input corpus (`input.txt`) and splits it into training (`train.bin`) and validation (`val.bin`) sets. Supports both character-level tokenization (default) and GPT-2 BPE (via `tiktoken`).
- `model.py`: Defines the GPT architecture, including the Transformer blocks, causal self-attention, LayerNorm, and weight tying between the embedding and the language modeling head.
- `train.py`: The training loop. Implements cosine learning rate decay with warmup, gradient clipping, AdamW optimizer with weight decay, and periodically saves checkpoints to the `out/` directory. Also generates text samples during training to monitor progress.
- `generate.py`: A script to load a trained checkpoint (`out/ckpt.pt`) and auto-regressively generate new text based on a prompt, supporting temperature and top-k sampling.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- `tiktoken` (optional, only required if using BPE tokenization)

## Usage

### 1. Prepare the Data
First, tokenize the dataset. By default, it uses character-level tokenization. This step creates `train.bin`, `val.bin`, and `meta.pkl`.
```bash
python prepare.py
```

### 2. Train the Model
Run the training script. The default configuration is tuned for a nano-sized model (~2M params). Checkpoints will be saved in the `out/` directory.
```bash
python train.py
```

### 3. Generate Text
Once you have a trained checkpoint, use the generation script to produce new text. You can customize the prompt, number of tokens, temperature, and top-k sampling.
```bash
python generate.py --prompt "Alice said" --tokens 300 --temp 0.8 --top_k 40
```

## Model Architecture Details

The model is a standard Causal Language Model using a Transformer decoder architecture:
- **Context Window (Block Size):** 256 tokens
- **Layers:** 6 Transformer blocks
- **Attention Heads:** 6
- **Embedding Dimension:** 384
- **Parameters:** ~2 Million
