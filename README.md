# tiny-lm

A character-level, decoder-only transformer language model built from scratch in PyTorch, as a learning project to understand how GPT-style models work internally with no `nn.Transformer`, no pretrained weights, every component (embeddings, positional encoding, attention, training loop, sampling) is hand-implemented. (Reference to: Andrej Karpathy)

## What's in here

- `tiny-genz.ipynb` — the full walkthrough: tokenization, batching, positional encoding, self-attention, training loop, and text generation.
- `input.txt` — training corpus (currently [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt); the project started out targeting a Gen-Z slang corpus, hence the notebook filename).

## Architecture

- **Tokenization**: character-level (`stoi`/`itos` lookup tables built from the corpus's unique characters).
- **Embeddings**: `nn.Embedding(vocab_size, d_embeddings)`, learned from scratch.
- **Positional encoding**: sinusoidal (`sin`/`cos`), added to token embeddings.
- **Self-attention**: custom multi-head attention (`Attention` class) with a causal mask (`torch.tril`) so the model can't peek at future tokens.
- **Output head**: linear projection from embedding space back to vocab-size logits.

This is a single-block model (one round of attention, no stacked transformer blocks, no LayerNorm/MLP yet) — a deliberately minimal starting point before adding depth.

## Usage

```bash
pip install torch jupyter
jupyter notebook tiny-genz.ipynb
```

Run the cells top to bottom to load the corpus, train `DecoderOnlyTransformer`, and generate text via the `generate()` function.

## Hyperparameters (current)

| Param | Value |
|---|---|
| `block_size` (context length) | 64 |
| `batch_size` | 128 |
| `d_embeddings` | 32 |
| `max_iters` | 5000 |
