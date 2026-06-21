## Mathematics

### Semantic Chunking

Given input sentences $S = [s_1, s_2, \dots, s_n]$ encoded to vectors:

$$E = \text{Encode}(S) = [\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_n], \quad \mathbf{e}_i \in \mathbb{R}^d$$

Cosine similarity between adjacent embeddings:

$$\text{sim}(i) = \cos(\mathbf{e}_{i-1},\, \mathbf{e}_i) = \frac{\mathbf{e}_{i-1} \cdot \mathbf{e}_i}{\|\mathbf{e}_{i-1}\|\,\|\mathbf{e}_i\|}$$

A new chunk boundary is inserted at position $i$ when similarity falls below threshold $T \in [0,1]$:

$$B = \bigl\{i \in \{2,\dots,n\} \mid \text{sim}(i) < T\bigr\}$$

The final chunks $C = [c_1, c_2, \dots, c_k]$ are contiguous sub-sequences of $S$ split at $B$:

$$c_j = \bigoplus_{i=b_{j-1}}^{b_j - 1} s_i$$

where $b_0 = 1$, $b_k = n+1$, $\{b_1, \dots, b_{k-1}\} = B$, and $\bigoplus$ denotes string concatenation. In compact form:

$$C = \text{Split}\!\left(S,\ \Bigl\{i \in \{2,\dots,n\} \;\Big|\; \frac{\mathbf{e}_{i-1}\cdot\mathbf{e}_i}{\|\mathbf{e}_{i-1}\|\,\|\mathbf{e}_i\|} < T\Bigr\}\right)$$

---

### Dense Retrieval — Maximal Marginal Relevance (MMR)

MMR balances **relevance** to the query against **diversity** among selected results. Given a query embedding $\mathbf{q}$ and a candidate set $\mathcal{D}$, let $\mathcal{R}$ be the set of already-selected documents. At each step, MMR selects:

$$d^* = \underset{d_i \in \mathcal{D} \setminus \mathcal{R}}{\arg\max} \Bigl[\lambda \cdot \cos(\mathbf{e}_i, \mathbf{q}) - (1 - \lambda) \cdot \max_{d_j \in \mathcal{R}} \cos(\mathbf{e}_i, \mathbf{e}_j)\Bigr]$$

where:
- $\lambda \in [0, 1]$ (`lambda_mult`) controls the relevance–diversity trade-off
- $\lambda = 1$ → pure similarity ranking (no diversity)
- $\lambda = 0$ → maximum diversity (relevance ignored)
- $\cos(\mathbf{e}_i, \mathbf{q})$ is the relevance of candidate $d_i$ to the query
- $\max_{d_j \in \mathcal{R}} \cos(\mathbf{e}_i, \mathbf{e}_j)$ is the redundancy of $d_i$ w.r.t. already-selected results

This is repeated $k$ times to produce the final top-$k$ results.

---

### Sparse Retrieval — BM25

BM25 (Best Match 25) ranks documents by term-frequency saturation and document-length normalisation. For a query $Q = \{q_1, \dots, q_m\}$ and document $d$:

$$\text{BM25}(d, Q) = \sum_{i=1}^{m} \text{IDF}(q_i) \cdot \frac{f(q_i, d) \cdot (k_1 + 1)}{f(q_i, d) + k_1 \cdot \left(1 - b + b \cdot \dfrac{|d|}{\text{avgdl}}\right)}$$

where:

$$\text{IDF}(q_i) = \ln\!\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)$$

| Symbol | Meaning |
|---|---|
| $f(q_i, d)$ | Term frequency of $q_i$ in document $d$ |
| $\|d\|$ | Length of document $d$ (in tokens) |
| $\text{avgdl}$ | Average document length across the corpus |
| $N$ | Total number of documents |
| $n(q_i)$ | Number of documents containing $q_i$ |
| $k_1$ | Term-frequency saturation parameter (default 1.5) |
| $b$ | Length normalisation parameter (default 0.75) |

BM25 excels at **exact keyword matching** and is complementary to dense retrieval, which handles **semantic paraphrase**.

---

### Hybrid Retrieval

ManuIndex fuses dense and sparse scores via a **weighted Reciprocal Rank Fusion** ensemble (LangChain `EnsembleRetriever`). The weight $\alpha$ controls the balance:

$$\text{score}_{\text{hybrid}}(d) = \alpha \cdot \text{score}_{\text{dense}}(d) + (1 - \alpha) \cdot \text{score}_{\text{sparse}}(d)$$

where:
- $\alpha \to 1$ → dense-only (semantic, MMR-ranked)
- $\alpha \to 0$ → sparse-only (lexical, BM25-ranked)
- $\alpha = 0.5$ → equal weight (default)

---

### Query Routing

At search time ManuIndex must identify which persisted FAISS index to load (one per document). It does this by comparing the **L2-normalised** query embedding against all stored summary embeddings using a dot product (equivalent to cosine similarity after normalisation):

$$\text{doc}^* = \underset{j}{\arg\max} \; \hat{\mathbf{q}} \cdot \hat{\mathbf{s}}_j$$

where $\hat{\mathbf{v}} = \mathbf{v} / \|\mathbf{v}\|$ and $\hat{\mathbf{s}}_j$ is the normalised summary embedding for document $j$.

---

