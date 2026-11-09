# Preserve the slow verifier; optimize only its matrix arithmetic

The first independent verifier completed every source/feature/prediction/metric
check and wrote its outputs, then failed the 180-second runtime assertion. A live
poll confirmed it was still running 233.32 seconds after registration. It exited
before a process-stop action was taken; no signal was sent. Its exact runtime and
peak RSS were not emitted because the receipt write followed the failing assertion.
Keep its source, execution receipt, failed log and all `results/qualified` outputs.

The slow step was a Python fsum for each feature times each projected coordinate
times each distinct input. Replace only that coordinate multiplication with a
NumPy vector-matrix product, still once per identical raw input, and retain the
independent feature construction and final scalar fsum. Compare every result with
the completed scalar reference to 1e-12 as well as the original predictions.
Keep the same 180-second / 4-GiB limits. Save the new output as `qualified-v2` and
the verifier as a separate frozen v2 snapshot. No training, model settings, data,
splits, selection, metrics definitions or precision changes.
