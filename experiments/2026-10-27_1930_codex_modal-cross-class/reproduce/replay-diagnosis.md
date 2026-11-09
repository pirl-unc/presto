# Training-probe cross-device replay diagnosis

The first fresh CPU float32 replay of shared-I-s42 failed the planned 2e-5
absolute-plus-relative criterion on 2/64 training probes, maximum absolute error
9.00773506e-5 log10(nM). No checkpoint, optimizer or holdout score changed.
Do not silently widen the threshold. Preserve this failure and compare the same
frozen checkpoint/probes under CPU float32, CPU without MKLDNN and CPU float64.
These are numerical diagnostics on training probes, not new predictive fits.
Record all per-probe differences before deciding whether a revised numerical
criterion is justified. The independent saved-prediction metric checks are separate.

CPU float32 and MKLDNN-disabled results are identical; CPU float64 differs from
CPU float32 by at most 2.5921e-7, but still differs from the saved GPU probes by
8.9952e-5. This localizes the discrepancy to a cross-device numerical path rather
than ordinary CPU accumulation error; the specific CUDA kernel remains unproven.
After the ten training jobs finish, run one sequential H100 verification worker
on the same volume/source snapshots, no optimizer or holdout evaluation. Replay
all ten saved 64-row training probes under original CUDA settings in fresh
processes. For the first reference only, compare CUDA float64 and disabled MHA
fastpath to isolate fused-kernel approximation. Preserve original tolerance and
all failed CPU receipts. This is verification of approved runs, not an extension
of the training comparison. Never exceed the approved two-H100 concurrency.

Closure: all ten original-H100 replays pass with zero error across640training probes. Disabling CUDA transformer fastpath moves the first reference to CPU agreement within5.12e-7; original CUDAfloat64 matches originalGPU within6.16e-7. Preserve strict CPU/altered-fastpath failures; no tolerance relaxation. See results/gpu-training-replay.json.
