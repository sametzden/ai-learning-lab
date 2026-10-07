# ShakespeareGPT

A small character-level GPT (decoder-only Transformer) written in PyTorch and trained on the Tiny Shakespeare dataset. Given a prompt such as `ROMEO:`, it continues the text one character at a time in a Shakespeare-like style.

This is a learning project built while following Atıl Samancıoğlu's course. The code is heavily commented on purpose: each file explains the *why* behind the step it implements (causal masking, Pre-LN, warmup, temperature, and so on).

## Model

| Setting | Value |
| --- | --- |
| Architecture | Decoder-only Transformer (GPT-style, Pre-LN) |
| Tokenization | Character-level, 65-symbol vocabulary |
| Layers | 6 |
| Attention heads | 6 |
| Embedding dimension | 384 |
| Context length (block size) | 256 characters |
| Dropout | 0.1 |
| Parameters | 10,795,841 |

Each block is masked multi-head self-attention followed by a 4x-expansion GELU MLP, both wrapped in residual connections with LayerNorm applied before the sublayer. Token and position embeddings are learned, and weights are initialized from N(0, 0.02).

## Project structure

```
dataset.py        Dataset download, character tokenizer, train/val split, batching
model.py          TransformerBlock and GPT (forward pass, loss, generate)
train.py          Training loop, LR warmup, periodic evaluation, checkpointing
generate.py       Loads a checkpoint and generates text (preset + interactive prompts)
data/             Tiny Shakespeare text (~1.1M characters)
requirements.txt  Python dependencies
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

For GPU training, install a CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

## Usage

Train the model:

```bash
python train.py
```

The dataset is downloaded to `data/shakespeare.txt` if it is missing. Train and validation loss plus a generated sample are printed every 500 iterations, and the final weights are saved to `checkpoints/model.pt`. Checkpoints are git-ignored, so you need to train before generating.

Generate text:

```bash
python generate.py
```

This prints completions for a few preset prompts and then opens an interactive prompt loop (`quit` to exit).

`train.py` uses CUDA when available and falls back to CPU. `generate.py` uses MPS when available and otherwise CPU.

## Training configuration

| Setting | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | 3e-4, linear warmup over the first 100 iterations, then constant |
| Batch size | 64 |
| Iterations | 5,000 |
| Gradient clipping | max norm 1.0 |
| Train / validation split | 90% / 10% |

## Results

Trained for 5,000 iterations on an NVIDIA RTX 4060 Ti. Losses are cross-entropy per character, averaged over 100 random batches:

| Split | Loss |
| --- | --- |
| Train | 0.33 |
| Validation | 2.21 |

The gap between the two shows that the model overfits: with ~10.8M parameters and only ~1M characters of training text, it memorizes much of the training set by iteration 5,000. Stopping earlier, keeping the checkpoint with the best validation loss, or raising dropout would generalize better.

Samples at temperature 0.8:

```
ROMEO:
As love as the house of Lancaster usurps,
Because sweeter than the Lord Northumberland.
Marshal, and think now the king shall be of discover?
And what more creature is now news with the crown?
And what may I say haze them now?
When they are abused by and by achieved by
The brother's gates, and, as I see them forth,
```

```
To be, or not to be druly secret,
I cannot boot for one. Come, let us go:
Despair of good us is the nights,
To take our neck nor swords; then in arms,
Our budden bear for breathing and begins to be
One of silence, and then stands in another
Degrees again; and one that life too,
```

## Acknowledgements

- Atıl Samancıoğlu's course, which this implementation follows
- [Tiny Shakespeare](https://github.com/karpathy/char-rnn) dataset by Andrej Karpathy
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (Radford et al., 2018)
