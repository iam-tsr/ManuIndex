## Mathematics

### Deterministic Neighbour Chunking

The current implementation uses a two-stage local-context algorithm: first, the document is split deterministically into ordered chunks; second, retrieved seed chunks are expanded with immediate neighbours while suppressing overlap.

Let a document be a character sequence $D$. Given an ordered separator list

$$\Sigma = [s_1, s_2, s_3]$$

with

$$s_1 = \text{double-newline}, \qquad s_2 = \text{newline}, \qquad s_3 = \text{space}$$

and a target chunk size $L$, recursive splitting produces an ordered list of raw pieces

$$P = [p_0, p_1, \dots, p_{m-1}] = \operatorname{Split}_{\Sigma, L}(D)$$

where each $p_i$ is a contiguous substring of $D$ obtained by recursively applying the separator priority until the chunk-length constraint is satisfied.

Whitespace-only pieces are discarded:

$$P' = [p_i \in P \mid \operatorname{strip}(p_i) \neq \varnothing]$$

The surviving pieces are assigned deterministic positional indices:

$$C = [(0, c_0), (1, c_1), \dots, (n-1, c_{n-1})]$$

where $c_i = p'_i$ and

$$\operatorname{chunk\_index}(c_i) = i, \qquad i \in \{0, \dots, n-1\}$$

This yields an index-to-text map

$$M : i \mapsto c_i$$

which is later used for local neighbour expansion.

Now let hybrid retrieval return an ordered list of seed chunks

$$R = [r_1, r_2, \dots, r_t]$$

and let $k(r_j)$ denote the deterministic chunk index attached to seed $r_j$ when present. The algorithm maintains a set $U$ of chunk indices that have already been emitted.

For a seed index $k$, define its radius-1 candidate window as

$$N(k) = \{k-1, k, k+1\} \cap \operatorname{dom}(M)$$

and the usable non-overlapping window as

$$W(k, U) = [i \in (k-1, k, k+1) \mid i \in \operatorname{dom}(M) \land i \notin U]$$

The emitted context for seed $r_j$ is then

$$E(r_j, U) =
\begin{cases}
r_j, & \text{if } k(r_j) \text{ is undefined} \\
\varnothing, & \text{if } k(r_j) \in U \\
\bigoplus_{i \in W(k(r_j), U)} c_i, & \text{if } W(k(r_j), U) \neq \varnothing
\end{cases}$$

and after emission the used-index set is updated by

$$U \leftarrow U \cup W(k(r_j), U)$$

So the full deterministic neighbour chunking procedure can be viewed as an ordered greedy cover over the deterministic chunk sequence:

$$\mathcal{E} = \operatorname{DNC}(D, R, \Sigma, L)$$

where $\operatorname{DNC}$ first constructs $C$ from $D$, then emits retrieval-conditioned radius-1 local windows in the order of $R$, subject to the non-overlap constraint

$$W(k_a, U_a) \cap W(k_b, U_b) = \varnothing \quad \text{for emitted windows } a \neq b$$

This preserves chunk order, keeps local context around each seed, and avoids duplicated output when adjacent seeds such as $4$, $5$, and $6$ are all retrieved.

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