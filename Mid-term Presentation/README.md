# FlashAttention-2

Citation: Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. Publisher/Conference. [https://crfm.stanford.edu/2023/07/17/flash2.html]

GitHub Repository: [https://crfm.stanford.edu/2023/07/17/flash2.html]

## Question 1: How to speed up the calculation for attention?

**Answer:** 

As the primary computational bottleneck in Transformer models is the softmax attention calculation, for GPT-3, the typical attention dimension is 128, the number of attention layers is 96, and the sequence length is 2048, the total number of FLOPs for attention is 128 * 96 * 2048 * 2048 = 2.6 * 10^14 (260 trillion). This is a huge number, and it takes a long time to compute. Therefore, it is important to speed up the calculation for attention.

- there are several ways to speed up the calculation for attention.

- First way is focusing on reducing the FLOP complexity. The FLOP complexity for sequences of length n is of O(n^2). Methods like Performers address this. For example, Performers employ a clever trick to approximate the softmax attention mechanism, reducing the complexity to linear O(n), which significantly speeds up the computation especially for long sequences.

- Although these methods reduce the compute requirements to linear or near-linear in sequence length, many of them do not display wall-clock speedup against standard attention and have not gained wide adoption. One main reason is that they focus on FLOP reduction (which may not correlate with wall-clock speed) and tend to ignore overheads from memory access (IO).

- Another way is to focus on reducing the latency. The latency for sequences of length n is of O(n). Methods like FlashAttention-2 address this. For example, FlashAttention-2 employs a clever trick to partition the work of the softmax attention mechanism, reducing the latency to logarithmic O(log(n)), which significantly speeds up the computation especially for long sequences.

- In this paper, we argue that a missing principle is making attention algorithms IO-aware [1]—that is, carefully accounting for reads and writes to different levels of fast and slow memory (e.g., between fast GPU on-chip SRAM and relatively slow GPU high bandwidth memory, or HBM

https://github.com/Dao-AILab/flash-attention/blob/main/assets/flashattn_banner.jpg?raw=true

## Question 2: