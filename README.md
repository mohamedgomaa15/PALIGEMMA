# 🦙 PaliGemma — Vision-Language Model (VLM) Implementation & Explanation

This repository contains a **fully-working re-implementation** of **PaliGemma** paper, Google’s **Vision-Language Model (VLM)** that combines:

- **Gemma (Large Language Model)** for autoregressive text generation  
- **SigLIP (Vision Encoder)** for extracting rich visual features

---

## This document provides a **technical deep dive** into the architecture, covering:

- Vision encoder details (Convolution, Learnable Positional Embeddings, Multi-Head Attention, Layer Norm, MLP, etc.)  
- Language decoder details (Embeddings, RoPE, Group-Query Attention, KV-Cache, RMSNorm, Gemma-MLP, Tie Weights, etc.)  
- Multimodal projection and embedding fusion  
- Autoregressive inference flow with example images and diagrams 

---

![PaliGemma Architecture](vlm.png)

---

## Model Overview

PaliGemma is a **vision-language model (VLM)** composed of:

1. A **SigLIP** vision encoder (ViT-style transformer)  
2. A **linear projection layer** to align visual embeddings with the language space  
3. A **Gemma** decoder (autoregressive Transformer)  

It takes both an image and text prompt as input, fuses them in embedding space, and produces text output (e.g., caption, question-answer, or reasoning). According to Google’s model card, the decoder is initialized from **Gemma-2B** and the vision encoder is SigLIP‑So400m. 

---

## Vision Encoder: SigLIP‑So400m

- The backbone is **SigLIP**, a Contrastive Vision Encoder.  
- Resolution: supports multiple input resolutions (224×224, 448×448, 896×896). 
- Patch embedding: input images are divided into **non‑overlapping patches**.  
- A **conv layer** projects each patch into a high-dimensional embedding space. 
- **Positional embeddings**: learned position embeddings are added to each patch embedding to encode spatial location. 
- **Transformer encoder layers**:  
  - Number of layers: According to the technical description, SigLIP‑So400m uses **27 encoder layers**. 
  - Each layer consists of:
    - Multi‑head self-attention (MHA) with *query, key, value* linear projections.
    - Layer normalization and **residual connections**.
    - A feed-forward MLP block with **GELU‑tanh activation**.  
  - Dropout/attention dropout is used during training (configurable).

 
---

## Multimodal Projection & Fusion

- After the vision encoder produces *per-patch embeddings* (shape: `[batch, num_patches, vision_hidden_dim]`), PaliGemma applies a **linear projection layer** (the “multi-modal projector”) to map these embeddings into the same dimensional space used by the Gemma language model.  
- A **scaling factor** is applied: image embeddings are scaled by \( \frac{1}{\sqrt{d_{\text{hidden}}}} \) before fusion. This stabilizes the magnitude when combining with text embeddings.  
- The fused sequence: image-derived embeddings are **injected into the token embedding stream** at positions corresponding to special `<image>` tokens, allowing the language model to treat image patches as additional “tokens.”

---

## Decoder: Gemma Language Model

The decoder is a **Gemma** autoregressive Transformer, initialized from Gemma-2B (in the canonical PaliGemma-3 B model). Here are key technical details:

### Multi‑Head Attention (GQA)

- Gemma uses a **generalized grouped‑query attention (GQA)** mechanism:
  - Instead of having as many key/value heads as query heads, Gemma **reduces the number of key/value heads** (num_key_value_heads). This reduces memory and computation for large models but retains expressivity.  
  - For example, with `num_attention_heads = H_q` and `num_key_value_heads = H_kv`, each key/value head is shared among `H_q / H_kv` query heads.  
- Attention computation:
  1. Project `hidden_states` into query, key, and value:
     \[
     Q = W_q \cdot h, \quad K = W_k \cdot h, \quad V = W_v \cdot h
     \]
  2. Reshape into multiple heads:
     \[
     Q: (B, H_q, L, d_{\text{head}}), \quad K,V: (B, H_kv, L, d_{\text{head}})
     \]
  4. **KV-Cache** for token-to-token generation, it adds the current `KV` to the cached `KVs` and returns the entire `KVs`.
  4. **Rotary positional embeddings** (RoPE) are applied to `V` and `K` (described below).
  5. Shared or repeated key/value heads: after rotation, Gemma **repeats** each `(K, V)` across groups to align with query heads. This is the “grouped query” trick.
  6. Attention weights are computed via scaled dot-product:
     \[
     \text{Attn} = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_{\text{head}}}} + \text{mask}\right)
     \]
  7. Output head concatenation and final projection with \( W_o \).

- **Causal masking** is used, because this is a decoder-only (autoregressive) model.

---

### Rotary Positional Embeddings (RoPE)

- Gemma uses **rotary embeddings** to encode relative position information(distance between tokens).
- A `GemmaRotaryEmbedding` module computes cosine and sine embeddings (per token position × head-dimension) that are applied elementwise to query and key tensors:
  - Given query vector \( q \) and key vector \( k \), and precomputed \( \cos(\cdot), \sin(\cdot) \), the rotated versions are:
    \[
    q_{\text{rot}} = q \cos + \text{rotate\_half}(q) \sin  
    \quad k_{\text{rot}} = k \cos + \text{rotate\_half}(k) \sin
    \]
  - This preserves length and injects relative position information without absolute positional embeddings.

---

### RMS Norm

- Rather than using the standard LayerNorm, Gemma uses **Root Mean Square Layer Normalization (RMSNorm)**:
  - RMSNorm normalizes by the root-mean-square (RMS) of the activations, rather than by mean and variance.
  - Specifically:  
    \[
    \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum x^2 + \epsilon}} \cdot (1 + w)
    \]
    where \( w \) is a learned parameter (dimension = hidden size) and \( \epsilon \) is a small constant.  
  - RMSNorm is lighter (fewer parameters) and has been shown to be effective in large-scale Transformer decoders.

---

### Feed‑Forward / MLP

- Each Transformer layer in Gemma includes a **gated MLP**:
  - There are **three linear projections**:
    - `up_proj`: from hidden size → intermediate size  
    - `gate_proj`: from hidden size → intermediate size  
    - `down_proj`: from intermediate size → hidden size  
  - Activation:  
    \[
    \text{MLP}(h) = \text{down\_proj} \Big( \text{GELU}(\text{gate\_proj}(h)) \;\times\; \text{up\_proj}(h) \Big)
    \]
    - The “gate” projection modulates the “up” projection via a non-linear activation → more expressive control.

---

### Tie‑Weights & Embedding Scaling

- The Gemma LM head shares its weights with the input embeddings: **tying** the weights ensures parameter efficiency and consistent embedding space.  

---

### KV‑Cache for Autoregressive Decoding

- To enable efficient generation, Gemma uses a **KV cache**:
  1. During generation, when producing token \( t \), you maintain a cache of *keys* and *values* from previous steps for each layer.
  2. On each new decoding step:
     - Compute query for the new token.
     - Retrieve / concat cached key / value tensors.
     - Compute attention only between new query and full key/value (cached + new), instead of recomputing everything.  
  3. This reduces time complexity from \( O(L^2) \) to \( O(L) \) per step (where \( L \) is generated length).

- The cache is usually stored as lists (one per layer), and updated on each step.
- KV-Cache caused memory-bound, so GQA is used for efficient KV-Cache

---

## Inference Flow

Here is a high‑level flow of how inference works in PaliGemma:

1. **Preprocessing**  
   - Input image is resized, normalized, and converted into patch embeddings by the SigLIP vision tower.  
   - Text prompt is tokenized. A sequence of `<image>` tokens is prepended so that the language model knows where to insert visual features.

2. **Embedding & Fusion**  
   - Vision tokens are projected into the decoder embedding space via a linear layer.  
   - Projected embeddings are scaled and *injected* at the positions of `<image>` tokens in the token sequence.  
   - Attention mask and positional IDs are constructed to allow proper causal decoding.

3. **Decoding**  
   - A `KVCache` object is initialized (empty).  
   - For the first step, input the full fused sequence (image + prompt) into the Gemma LLM.  
   - On subsequent steps:
     - Pass only the newly generated token (or small chunk).  
     - Use the `KVCache` to attend over previous keys/values efficiently.  
   - Apply top-p (nucleus) sampling or greedy decoding based on your inference strategy.

4. **Post‑processing**  
   - Decode token ids back into strings, skipping special tokens.  
   - Optionally, you can translate or format the output, depending on use-case.

---

## Architecture Diagram

Below is a block-diagram (conceptual) of the PaliGemma architecture:

```text
+---------------------+      +---------------------+      +---------------------+
|     Input Image      | -->  |  SigLIP Vision Tower | --> |  Vision Embeddings  |
+---------------------+      +---------------------+      +---------------------+
                                                                 |
                                                                 v
                                                +------------------------------+
                                                |   Linear Projection / Scale   |
                                                +------------------------------+
                                                                 |
                                                                 v
+--------------------+    +---------------------+    +-----------------------+
|  Text Prompt Tokens | -> | Gemma Token Embeds | -> | Embedding Fusion (text + image) |
+--------------------+    +---------------------+    +-----------------------+
                                                                 |
                                                                 v
+-------------------------------------------------------------------------------+
|                              Gemma Decoder (Autoregressive)                    |
|                                                                                |
|   - Multi‑Head Attention (GQA) + RoPE                                          |
|   - RMSNorm Layers                                                             |
|   - Gated MLP with GELU‑Tanh                                                   |
|   - Residual Connections                                                       |
|   - KV-Cache for efficient decoding                                            |
+-------------------------------------------------------------------------------+
                  |
                  v
+-------------------------------------+
|      Output LM Head / Logits        |
+-------------------------------------+
```
---

## References & Further Reading

- [PaliGemma paper (v1) — PaliGemma: A versatile 3B VLM for transfer](https://arxiv.org/abs/2407.07726?utm_source=chatgpt.com)
- [PaliGemma 2 paper — PaliGemma 2: A Family of Versatile VLMs](https://arxiv.org/abs/2412.03555?utm_source=chatgpt.com)
- [PaLI‑3 Model (training recipe inspiration)](https://arxiv.org/abs/2310.09199?utm_source=chatgpt.com)
- [Technical blog on PaliGemma / Gemma architecture ](https://developers.googleblog.com/gemma-explained-paligemma-architecture/?utm_source=chatgpt.com)

