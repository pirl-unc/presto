# Joint MHC-I/II multi-output model that beats NetMHCpan BA and NetMHCIIpan — plan

Agent: claude. Date: 2026-10-30. Status: proposed; nothing launched.
Builds on #131 (shared-v7: one encoder, groove interaction, register scoring shared by class I/II),
issues #138-#142. Success is defined before any new fit.

## Goal and honest starting point
One model, one shared representation and decoder, class I and II BA (IC50/KD/EC50) outputs, whose
held-out within-assay/study ranking and thresholded binding metrics beat NetMHCpan-4.1 BA (class I)
and NetMHCIIpan-4.3 BA (class II) on identical peptides/MHCs. Today's test Spearman is ~.1-.3 per
assay (PR #131 table), with no external comparison yet. The gap is unmeasured, so measurement comes first.

## Success criterion (pre-registered)
On a held-out set with tool-training overlap audited out: per class, per assay/study, Spearman and
AUROC at <=500 nM with bootstrap CIs over studies/peptides; win = lower CI bound of
(Presto - tool) > 0 on the pooled-within-study metric, with no class regressing vs the earlier Presto
checkpoint. Aggregates across assay groups never count (see the assay-only .636 diagnostic).

## PR series (each stacked on #131; one concern per PR)
1. **External benchmark harness (#140), CPU only.** Pinned local tools (netMHCpan 4.1, netMHCIIpan 4.3,
   mhcflurry 2.3.16 under ~/code/netmhc-bundle and the shared venv). Identical-population prediction
   dumps, prospective multi-allelic/unsupported-allele rules, per-study metrics, assay-only/train-mean
   baselines fit without held-out labels, overlap audit vs each tool's training data.
2. **Score existing checkpoints on it** (direct-factorized s42/s43, curated s43). Quantifies the gap per
   class/assay. Decides which later PR matters.
3. **MHC-II coverage reconciliation (#139).** Source/curation waterfall vs the NetMHCIIpan-4.3 corpus.
4. **Training experiments (#141).** Exposure, LR schedule, within-assay pairwise ranking loss, one change
   per arm, matched compute, 2 seeds.
5. **Full-epoch shared model with the best recipe.** H100!, <=2 workers, registered.
6. Optional: #138 stability on the shared decoder, #142 synthetic prior. Not on the BA critical path.

## Cost/approval
PRs 1-3 are local CPU/read-only. PRs 4-5 need Modal H100! runs; confirm before launching.
Tool outputs are third-party licensed predictions; keep dumps in artifacts/, hashes in git.
