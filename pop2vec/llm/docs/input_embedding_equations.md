# Input Embedding Equations & Transformations

This document summarizes the mathematical operations and equations used to process each dimension of the `input_ids` tensor in your transformer model, with detailed explanations of the Embedding and ReZero functions.

---

## 1. **Input Structure**

Each input sequence is a tensor of shape `[N, 4, 512]`:
- **N**: Number of sequences (batch size)
- **4**: Feature dimensions per token:
  1. Event Token ID ($t_i$)
  2. Absolute Position (Days Since Genesis, $p_i$)
  3. Age ($a_i$)
  4. Segment/Continuation Flag ($s_i$)
- **512**: Sequence length (max tokens per sequence)

---

## 2. **Token Embedding**

For each token position $i$:
- **Event Token ID**: $t_i$
- Embedded via a lookup:
  $$
  \mathbf{e}_i = \text{Embedding}_{\text{token}}(t_i)
  $$
  where $\mathbf{e}_i \in \mathbb{R}^{d}$ and $d$ is the hidden size.

**What happens in Embedding function?**
- The embedding function is a lookup table: for each discrete token ID $t_i$, it returns a learnable vector $\mathbf{e}_i$.
- Formally, if $E \in \mathbb{R}^{V \times d}$ is the embedding matrix ($V$ = vocab size), then:
  $$
  \mathbf{e}_i = E[t_i]
  $$
- The embedding matrix $E$ is initialized randomly and learned during training.

---

## 3. **Positional Embedding (Time2Vec)**

### **a. Age Embedding**

Given age $a_i$ for token $i$:
- Embedded using Time2Vec with cosine activation:
  $$
  \mathbf{p}^{(\text{age})}_i = \text{Time2Vec}_{\cos}(a_i)
  $$
  where
  $$
  \text{Time2Vec}_{f}(x) = \left[ f(w_1 x + b_1), \ldots, f(w_{d-1} x + b_{d-1}), w_0 x + b_0 \right]
  $$
  - $f$ is $\cos$ for age
  - $w_j, b_j$ are learnable parameters

### **b. Absolute Position Embedding**

Given absolute position $p_i$ for token $i$:
- Embedded using Time2Vec with sine activation:
  $$
  \mathbf{q}^{(\text{abs})}_i = \text{Time2Vec}_{\sin}(p_i)
  $$
  where
  $$
  \text{Time2Vec}_{f}(x) = \left[ f(w_1 x + b_1), \ldots, f(w_{d-1} x + b_{d-1}), w_0 x + b_0 \right]
  $$
  - $f$ is $\sin$ for absolute position

**Clarification on Notation:**  
- $a_i$ is the age input, $\mathbf{p}^{(\text{age})}_i$ is the output embedding for age.
- $p_i$ is the absolute position input, $\mathbf{q}^{(\text{abs})}_i$ is the output embedding for absolute position.
- The output is always a vector of dimension $d$.

---

## 4. **Segment Embedding**

For segment/continuation flag $s_i$:
- Embedded via a lookup:
  $$
  \mathbf{e}^{(\text{seg})}_i = \text{Embedding}_{\text{segment}}(s_i)
  $$
- Like token embedding, this is a learnable lookup table for each possible segment value.

---

## 5. **Combining Embeddings**

For each token position $i$, the final embedding is:
- **Residual connections (ReZero):**
  $$
  \mathbf{h}_i = \mathbf{e}_i + \alpha_1 \cdot \mathbf{p}^{(\text{age})}_i + \alpha_2 \cdot \mathbf{q}^{(\text{abs})}_i + \alpha_3 \cdot \mathbf{e}^{(\text{seg})}_i
  $$
  - $\alpha_j$ are learnable scalar weights (from ReZero blocks).

**What happens in ReZero function?**
- ReZero is a residual connection with a learnable scalar gate:
  $$
  \text{ReZero}(x, y) = x + \alpha \cdot y
  $$
  - $\alpha$ is initialized to zero and learned during training.
  - This allows the network to start as an identity mapping and gradually learn to use the additional information.

- In your code, you apply ReZero sequentially:
  1. Add age embedding to token embedding: $\mathbf{e}_i \leftarrow \text{ReZero}(\mathbf{e}_i, \mathbf{p}^{(\text{age})}_i)$
  2. Add absolute position embedding: $\mathbf{e}_i \leftarrow \text{ReZero}(\mathbf{e}_i, \mathbf{q}^{(\text{abs})}_i)$
  3. Add segment embedding: $\mathbf{e}_i \leftarrow \text{ReZero}(\mathbf{e}_i, \mathbf{e}^{(\text{seg})}_i)$

- The final embedding is then passed through dropout:
  $$
  \mathbf{h}_i = \text{Dropout}(\mathbf{e}_i)
  $$

---

## 6. **Time2Vec Equation (Full)**

For a scalar input $\tau$ (age or position), output dimension $d$:
- Learnable parameters: $w_0, b_0 \in \mathbb{R}$, $w \in \mathbb{R}^{d-1}$, $b \in \mathbb{R}^{d-1}$
- Nonlinear part:
  $$
  v_1 = f(\tau w + b)
  $$
- Linear part:
  $$
  v_2 = \tau w_0 + b_0
  $$
- Concatenated output:
  $$
  \text{Time2Vec}(\tau) = [v_1, v_2]
  $$

---

## 7. **Summary Table**

| Dimension         | Input Symbol | Output Symbol                | Embedding Equation                                 | Activation |
|-------------------|-------------|------------------------------|---------------------------------------------------|------------|
| Event Token ID    | $t_i$       | $\mathbf{e}_i$               | $\mathbf{e}_i = \text{Embedding}_{\text{token}}(t_i)$ | -          |
| Age               | $a_i$       | $\mathbf{p}^{(\text{age})}_i$| $\mathbf{p}^{(\text{age})}_i = \text{Time2Vec}_{\cos}(a_i)$ | $\cos$     |
| Abs. Position     | $p_i$       | $\mathbf{q}^{(\text{abs})}_i$| $\mathbf{q}^{(\text{abs})}_i = \text{Time2Vec}_{\sin}(p_i)$ | $\sin$     |
| Segment           | $s_i$       | $\mathbf{e}^{(\text{seg})}_i$| $\mathbf{e}^{(\text{seg})}_i = \text{Embedding}_{\text{segment}}(s_i)$ | -          |

---

## 8. **References**

- [Time2Vec: Learning a Vector Representation of Time](https://arxiv.org/abs/1907.05321)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [ReZero: Residual Connections Without Normalization](https://arxiv.org/abs/2003.04887)

---

**This document should help you understand how each input dimension is mathematically processed in your model, with details on Embedding and ReZero functions.**