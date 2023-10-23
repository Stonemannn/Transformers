# FlashAttention-2 

Citation: Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. [https://crfm.stanford.edu/2023/07/17/flash2.html]

GitHub Repository: [https://crfm.stanford.edu/2023/07/17/flash2.html]

## Overview
The attention layer is the main bottleneck in scaling Transformers to longer sequences. Its runtime and memory increase quadratically with sequence length. It is important to speed up the calculation for attention.

### Question 1: How to speed up the calculation for attention?

**Answer:** 
- There are several ways to speed up the calculation for attention.

- This paper is making attention algorithms IO-aware —that is, accounting for reads and writes to different levels of fast and slow memory (e.g., between fast GPU on-chip SRAM and relatively slow GPU high bandwidth memory, or HBM)

- **How FlashAttention Works**: It reorders the attention computation and uses classical techniques like tiling and recomputation to significantly speed up and reduce memory usage.

![Flash Attention Banner](https://github.com/Dao-AILab/flash-attention/blob/main/assets/flashattn_banner.jpg?raw=true)

## Psuedocodes for standard attention & FlashAttention are shown below.

**Algorithm 0: Standard Attention**
![standard attention](https://github.com/Stonemannn/Transformers/blob/36f93dcb69ba4d846f44c1082fab93d1c901ef00/Mid-term%20Presentation/figures/standard_attention.png?raw=true)

**Algorithm 1: FlashAttention**
![flash attention](https://github.com/Stonemannn/Transformers/blob/36f93dcb69ba4d846f44c1082fab93d1c901ef00/Mid-term%20Presentation/figures/FlashAttention.png?raw=true)

- **Speedup**: FlashAttention achieves 2-4× wall-clock time speedup over standard attention.

### Question 2: Is FlashAttention perfect? Is there any other way to speed up FlashAttention?

**Answer:**
- FlashAttention-2 is an improved version of FlashAttention. It aims to address the inefficiencies in FlashAttention by better work partitioning between different thread blocks and warps on the GPU.

![partitioning](https://github.com/Stonemannn/Transformers/blob/36f93dcb69ba4d846f44c1082fab93d1c901ef00/Mid-term%20Presentation/figures/flash_flash2_partitioning.png?raw=true)

Let's consider a simplified example to illustrate the difference between FlashAttention and FlashAttention-2 in terms of shared memory usage. Assume we have $Q$, $K$, and $V$ matrices, and we are using 4 warps (W1, W2, W3, W4) for computation.

### FlashAttention:

1. **Splitting**: $K$ and $V$ are split across 4 warps. Let's say $K_1$, $K_2$, $K_3$, $K_4$ are the slices of $K$ assigned to W1, W2, W3, W4 respectively.
  
2. **Computation**: Each warp calculates a slice of $QK^T$ using its slice of $K$. For example, W1 calculates $QK_1^T$, W2 calculates $QK_2^T$, and so on.
  
3. **Intermediate Results**: Now, each warp has a partial result of $QK^T$ that needs to be combined to get the full $QK^T$ matrix. 

4. **Shared Memory**: All warps write their partial results to shared memory.

5. **Synchronization**: Warps synchronize and read from shared memory to add up these partial results to get the full $QK^T$.

### FlashAttention-2:

1. **Splitting**: $Q$ is split across 4 warps. Let's say $Q_1$, $Q_2$, $Q_3$, $Q_4$ are the slices of $Q$ assigned to W1, W2, W3, W4 respectively. $K$ and $V$ are accessible to all warps.
  
2. **Computation**: Each warp calculates a slice of $QK^T$ using its slice of $Q$. For example, W1 calculates $Q_1K^T$, W2 calculates $Q_2K^T$, and so on.
  
3. **Intermediate Results**: There's no need to combine different slices of $QK^T$.

## Psuedocodes for FlashAttention & FlashAttention-2 are shown below.
**Algorithm 1: FlashAttention**
![flash attention](https://github.com/Stonemannn/Transformers/blob/36f93dcb69ba4d846f44c1082fab93d1c901ef00/Mid-term%20Presentation/figures/FlashAttention.png?raw=true)

**ALgorithm 2: FlashAttention-2**
![flash attention2](https://github.com/Stonemannn/Transformers/blob/36f93dcb69ba4d846f44c1082fab93d1c901ef00/Mid-term%20Presentation/figures/FlashAttention2.png?raw=true)
---


- **Speedup**: FlashAttention-2 yield around 2× speedup compared to FlashAttention

![speedup](https://github.com/Stonemannn/Transformers/blob/36f93dcb69ba4d846f44c1082fab93d1c901ef00/Mid-term%20Presentation/figures/flash2_a100_fwd_bwd_benchmark.png?raw=true)

### Critical Analysis
- **Future Applications**: FlashAttention-2 makes it possible for video generation which require long sequences.


## Resource links
1. OpenAI. Gpt-4 technical report.https://arxiv.org/pdf/2303.08774.pdf
2. Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems.https://arxiv.org/pdf/2205.14135.pdf
3. Noam Shazeer. Fast transformer decoding: One write-head is all you need. https://arxiv.org/abs/1911.02150
4. Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints.https://arxiv.org/pdf/2305.13245.pdf
5. Zhe Jia, Blake Tillman, Marco Maggioni, and Daniele Paolo Scarpazza. Dissecting the graphcore IPU architecture via microbenchmarking.https://arxiv.org/abs/1912.03413

## References

- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
Paper: https://arxiv.org/abs/2205.14135

- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
Tri Dao
Paper: https://tridao.me/publications/flash2/flash2.pdf

