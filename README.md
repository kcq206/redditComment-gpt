# TinyGPT — Training a Transformer Language Model from Scratch

A minimal GPT-style language model built in **PyTorch** and trained on Reddit comments.  
This project demonstrates how transformer-based language models work internally by implementing the core components of GPT architecture without relying on high-level libraries like HuggingFace.

The goal of this project is to better understand **tokenization, transformer blocks, attention mechanisms, and next-token prediction** by building and training a model from the ground up.

---

# Project Overview

This project implements a simplified version of a GPT language model capable of generating text by predicting the next token in a sequence.

The model learns language by training on a dataset of Reddit comments and optimizing a **next-token prediction objective**.

Key components implemented include:

- Byte Pair Encoding (BPE) tokenizer
- Token embeddings
- Positional embeddings
- Multi-head self-attention
- Transformer blocks
- Layer normalization
- Residual connections
- Feed-forward networks
- Autoregressive training loop

---


### Model Hyperparameters

| Parameter | Value |
|---|---|
| Context Length | 256 tokens |
| Embedding Size | 384 |
| Transformer Layers | 6 |
| Attention Heads | 6 |
| Batch Size | 16 |
| Vocabulary Size | 16,000 |

---

# Key Concepts Implemented

### Tokenization
Text is converted into tokens using **Byte Pair Encoding (BPE)**.



The model repeatedly predicts the next token and appends it to the sequence.

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Byte Pair Encoding Tokenization

---

# What I Learned

This project helped deepen my understanding of:

- How transformer architectures work internally
- The mechanics of self-attention
- Tokenization and vocabulary design
- Training autoregressive language models
- PyTorch model implementation and optimization
- Text generation strategies (sampling, temperature, top-k)

---

# Future Improvements

Potential improvements to this project include:

- Implementing **FlashAttention for faster training**
- Adding **Top-K / Top-P sampling for generation**
- Training on a **larger dataset**
- Increasing model depth and embedding size
- Implementing **custom attention from scratch**

---

# Running the Project

### Train the Model
python train.py

The trained model will be saved as:
model_tinygpt.pt


---

# Why I Built This

I built this project to gain a deeper understanding of transformer-based language models by implementing the core architecture directly in PyTorch rather than relying solely on prebuilt libraries.
