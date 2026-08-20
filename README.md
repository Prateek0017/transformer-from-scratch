# GPT Transformer from Scratch

A PyTorch implementation exploring the architecture and building blocks of
GPT-style Transformer language models. Written to understand attention,
positional encoding, and causal language modeling by building each component
rather than importing them from a library.

> Originally developed in 2023 as a deep-learning coding assignment.
> Preserved here as an early exploration of Transformer internals.

## What's implemented

Across three notebooks, the following components are built and tested:

- GPT-style model configuration (`GPT1Config`)
- Token embeddings + positional representations
- Multi-head self-attention with causal masking
- Transformer blocks (attention → LayerNorm → feed-forward → residual)
- Language modeling head
- Forward-pass validation on random input sequences

Deliberately avoids importing `GPT2Model` or similar high-level classes —
the goal was to construct the architecture manually, not use it.

## Model configuration

| Parameter | Value |
|---|---|
| Vocabulary size | 10,000 |
| Context length | 512 |
| Embedding dimension | 768 |
| Attention heads | 12 |
| Transformer blocks | 12 |

## Notebooks

- `task-1.ipynb` — core Transformer implementation and forward-pass validation
- `task-2.ipynb` — architectural variants and attention modifications
- `task-3.ipynb` — distributed training exploration (DDP / FSDP)

Kaggle-hosted versions of each:

- [Task 1 on Kaggle](https://www.kaggle.com/code/prateekupadhayay/task-1)
- [Task 2 on Kaggle](https://www.kaggle.com/code/prateekupadhayay/task-2)
- [Task 3 on Kaggle](https://www.kaggle.com/code/prateekupadhayay/task-3)

A `SUMMARY of all task.pdf` document is included with methodology notes for
each task.

## Verified output

Forward pass on a random input sequence of length 512 produces the expected
logits tensor:

​```
Output shape: torch.Size([1, 512, 10000])
​```

---

Matches `(batch_size, sequence_length, vocab_size)` as expected for a
language modeling head.

## What this project is not

Being explicit so the scope is clear:

- Not trained on real text. Forward-pass verified on random tokens only.
- Not benchmarked against reference implementations.
- Not optimized for training efficiency (no mixed precision, no gradient
  checkpointing, no fused kernels).
- Uses standard multi-head attention rather than more recent variants
  (grouped-query, sliding-window, etc.) — those are discussed in
  `task-2.ipynb` but not integrated into the base model.

For production Transformer work I would use HuggingFace `transformers` or
PyTorch's native modules. This repository is an educational implementation.

## Tech stack

Python, PyTorch, Jupyter

## References

- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2)
- Andrej Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT)
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models*
- Beltagy et al., *Longformer: The Long-Document Transformer*
- PyTorch [DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- PyTorch [FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

