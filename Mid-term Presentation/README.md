# FlashAttention-2 

Citation: Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. Publisher/Conference. [https://crfm.stanford.edu/2023/07/17/flash2.html]

GitHub Repository: [https://crfm.stanford.edu/2023/07/17/flash2.html]


Today I'll be presenting a paper that addresses the challenges of speeding up attention mechanisms in Transformers. The paper is titled "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" by Tri Dao. I will only focus on Forward Pass Attention in this presentation.

As the primary computational bottleneck in Transformer models is the softmax attention calculation, for GPT-3, the typical attention dimension is 128, the number of attention layers is 96, and the sequence length is 2048, the total number of FLOPs for attention is 128 * 96 * 2048 * 2048 = 2.6 * 10^14 (260 trillion). This is a huge number, and it takes a long time to compute. Therefore, it is important to speed up the calculation for attention.

## Question 1: How to speed up the calculation for attention?

**Answer:** 
- there are several ways to speed up the calculation for attention.

- First way is focusing on reducing the FLOP complexity. The FLOP complexity for sequences of length n is of O(n^2). Methods like Performers approximate the softmax attention mechanism, reducing the complexity to linear O(n). However, these methods do not display wall-clock speedup against standard attention and have not gained wide adoption. In fact, on modern GPUs, compute speed has out-paced memory speed, and most operations in Transformers are bottlenecked by memory accesses.

- This paper is making attention algorithms IO-aware —that is, carefully accounting for reads and writes to different levels of fast and slow memory (e.g., between fast GPU on-chip SRAM and relatively slow GPU high bandwidth memory, or HBM

![Flash Attention Banner](https://github.com/Dao-AILab/flash-attention/blob/main/assets/flashattn_banner.jpg?raw=true)


## Question 2: Is Flash Attention perfect? Is there any other way to speed up the Flash Attention?

**Answer:**

Let's consider a simplified example to illustrate the difference between FlashAttention and FlashAttention-2 in terms of shared memory usage. Assume we have a $Q$, $K$, and $V$ matrix, and we are using 4 warps (W1, W2, W3, W4) for computation.

### FlashAttention:

1. **Splitting**: $K$ and $V$ are split across 4 warps. Let's say $K_1$, $K_2$, $K_3$, $K_4$ are the slices of $K$ assigned to W1, W2, W3, W4 respectively.
  
2. **Computation**: Each warp calculates a slice of $QK^T$ using its slice of $K$. For example, W1 calculates $QK_1^T$, W2 calculates $QK_2^T$, and so on.
  
3. **Intermediate Results**: Now, each warp has a partial result of $QK^T$ that needs to be combined to get the full $QK^T$ matrix. 

4. **Shared Memory**: All warps write their partial results to shared memory.

5. **Synchronization**: Warps synchronize and read from shared memory to add up these partial results to get the full $QK^T$.

6. **Final Step**: Each warp then multiplies its slice of $QK^T$ with its slice of $V$ to get the final output.

### FlashAttention-2:

1. **Splitting**: $Q$ is split across 4 warps. Let's say $Q_1$, $Q_2$, $Q_3$, $Q_4$ are the slices of $Q$ assigned to W1, W2, W3, W4 respectively. $K$ and $V$ are accessible to all warps.
  
2. **Computation**: Each warp calculates a slice of $QK^T$ using its slice of $Q$. For example, W1 calculates $Q_1K^T$, W2 calculates $Q_2K^T$, and so on.
  
3. **Intermediate Results**: Each warp directly multiplies its slice of $QK^T$ with $V$ to get its part of the output. There's no need to combine different slices of $QK^T$ because each warp is responsible for a distinct part of the output.

4. **Shared Memory**: No need to write to or read from shared memory for combining intermediate results.

5. **Synchronization**: No synchronization needed among warps for this step.

6. **Final Step**: Each warp already has its slice of the final output.

```latex
\begin{algorithm}
\caption{FlashAttention-2 forward pass}
\begin{algorithmic}[1]
\Require Matrices \( Q, K, V \in \mathbb{R}^{A \times A} \) in HBM, block sizes \( A_1, A_2 \).
\State Divide \( Q \) into \( A_1 \times A_2 \) blocks \( Q_1, \ldots, Q_{A_1 \times A_2} \) of size \( A_1 \times A_2 \) each, and divide \( K, V \) into \( A_1 \times A_2 \) blocks \( K_1, \ldots, K_{A_1 \times A_2} \) and \( V_1, \ldots, V_{A_1 \times A_2} \), of size \( A_1 \times A_2 \) each.
\State Divide the output \( O \in \mathbb{R}^{A \times A} \) into \( A_1 \times A_2 \) blocks \( O_1, \ldots, O_{A_1 \times A_2} \) of size \( A_1 \times A_2 \) each.
\For{\( 1 \leq i \leq A_1 \times A_2 \)}
    \State Load \( Q_i \) from HBM to on-chip SRAM.
    \State On chip, initialize \( O_i^{(0)} = \mathbf{0} \), \( \ell_i^{(0)} = \mathbf{0} \), \( A_i^{(0)} = -\infty \).
    \For{\( 1 \leq j \leq A_1 \times A_2 \)}
        \State Load \( K_j, V_j \) from HBM to on-chip SRAM.
        \State On chip, compute \( S_i^{(j)} = Q_i K_j \).
        \State On chip, compute \( A_i^{(j)} = \max(A_i^{(j-1)}, \text{rowmax}(S_i^{(j)})) \).
        \State On chip, compute \( \ell_i^{(j)} = \ldots \)  % truncated for brevity
        \State On chip, compute \( O_i^{(j)} = \ldots \)  % truncated for brevity
    \EndFor
    \State On chip, compute \( O_i = \ldots \)  % truncated for brevity
    \State On chip, compute \( A_i = \ldots \)  % truncated for brevity
    \State Write \( O_i \) to HBM as the \( i \)-th block of \( O \).
    \State Write \( A_i \) to HBM as the \( i \)-th block of \( A \).
\EndFor
\State Return the output \( O \) and the logsumexp \( A \).
\end{algorithmic}
\end{algorithm}

```
