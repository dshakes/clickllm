# Proving fused-rmsnorm

**Claim:** fused-rmsnorm: claims 1.18× · drift expected: fp16 reduction order

Loading is the easy half. These steps are the half that decides whether
the kernel should ship.

1. capture or reuse an eval set from real traffic — a kernel that is correct on a benchmark and wrong on your prompts is the failure this whole loop exists to catch
2. run the eval set against **stock** and record a receipt; that digest is the baseline the kernel is compared against, and it must not be regenerated afterwards
3. the author expects drift (fp16 reduction order), so equivalence is a statistical question: run `clickllm prove` and require the interval's lower bound to clear the bar, not the point estimate
4. measure throughput at YOUR concurrency, not at batch 1. Kernel speedups quoted single-stream routinely vanish under real batching, which is the same trap speculative decoding sets
5. compare against the claim: 1.18×. A kernel that delivers materially less than claimed is not a win to be rounded up, it is a result to be reported
6. keep the receipt. When the kernel is rebuilt against a newer vLLM, the guard's fingerprint check tells you the proof no longer applies

> A kernel that is 1.4× faster and changes one logit in ten thousand is
> not a 1.4× win. It is an unreviewed model change wearing a benchmark.
