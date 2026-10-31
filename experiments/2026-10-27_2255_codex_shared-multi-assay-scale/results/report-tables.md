# Independently verified comparison tables

Two-seed means of individual-model metrics, not ensemble predictions.
IC50 comparisons use identical original held-out observation IDs.
RMSE is log10(nM), except half-life in log10(minutes).

## Original IC50 cohort

| Condition | I validation ρ / RMSE | I test ρ / RMSE | II validation ρ / RMSE | II test ρ / RMSE |
|---|---:|---:|---:|---:|
| ic50-exact | 0.6487 / 1.0516 | 0.6025 / 1.1100 | 0.6511 / 0.7878 | 0.5893 / 0.7618 |
| ic50-bounds | 0.6586 / 1.0455 | 0.6022 / 1.1048 | 0.6500 / 0.8142 | 0.6023 / 0.7677 |
| multi-assay | 0.6555 / 1.0430 | 0.6129 / 1.0978 | 0.5750 / 0.8560 | 0.5448 / 0.8122 |
| multi-assay-wide | 0.6800 / 1.0094 | 0.6407 / 1.0702 | 0.5966 / 0.8456 | 0.5679 / 0.7824 |

## Six-output results

n counts exact observations. All bounds contribute to loss; classification,
Pearson and mean-baseline metrics are in the machine-readable tables.
Tiny or missing strata do not support generalization claims.

| Condition | Output | Class | Validation n | Validation ρ / RMSE | Test n | Test ρ / RMSE | Test train-mean RMSE |
|---|---|---|---:|---:|---:|---:|---:|
| multi-assay | ic50 | I | 1968 | 0.6555 / 1.0430 | 1758 | 0.6129 / 1.0978 | 1.4433 |
| multi-assay | ic50 | II | 758 | 0.5750 / 0.8560 | 842 | 0.5448 / 0.8122 | 1.1008 |
| multi-assay | kd | I | 567 | 0.5337 / 1.4104 | 561 | 0.4679 / 1.4848 | 1.2753 |
| multi-assay | kd | II | 32 | 0.1107 / 1.0269 | 82 | 0.1543 / 1.0720 | 1.5684 |
| multi-assay | ec50 | I | 31 | 0.2364 / 1.2782 | 47 | 0.3681 / 1.1020 | 1.1920 |
| multi-assay | ec50 | II | 3 | 0.0000 / 0.9013 | 0 | — / — | — |
| multi-assay | kd_via_ic50 | I | 5182 | 0.6811 / 1.0668 | 4808 | 0.6757 / 1.0611 | 1.4106 |
| multi-assay | kd_via_ic50 | II | 183 | 0.5183 / 0.8639 | 123 | 0.1982 / 0.9028 | 1.1113 |
| multi-assay | kd_via_ec50 | I | 1175 | 0.5308 / 1.2923 | 1154 | 0.5412 / 1.2607 | 1.2101 |
| multi-assay | kd_via_ec50 | II | 1529 | 0.5770 / 0.7468 | 1231 | 0.5237 / 0.7330 | 0.8433 |
| multi-assay | half_life | I | 968 | 0.6876 / 0.8734 | 934 | 0.6620 / 0.8661 | 1.1847 |
| multi-assay | half_life | II | 0 | — / — | 2 | 1.0000 / 1.6800 | 2.2367 |
| multi-assay-wide | ic50 | I | 1968 | 0.6800 / 1.0094 | 1758 | 0.6407 / 1.0702 | 1.4433 |
| multi-assay-wide | ic50 | II | 758 | 0.5966 / 0.8456 | 842 | 0.5679 / 0.7824 | 1.1008 |
| multi-assay-wide | kd | I | 567 | 0.5815 / 1.3333 | 561 | 0.5586 / 1.3477 | 1.2753 |
| multi-assay-wide | kd | II | 32 | 0.4536 / 0.8804 | 82 | 0.2908 / 1.0465 | 1.5684 |
| multi-assay-wide | ec50 | I | 31 | 0.4654 / 1.1749 | 47 | 0.1853 / 1.1826 | 1.1920 |
| multi-assay-wide | ec50 | II | 3 | 0.0000 / 0.8999 | 0 | — / — | — |
| multi-assay-wide | kd_via_ic50 | I | 5182 | 0.7063 / 1.0342 | 4808 | 0.6999 / 1.0345 | 1.4106 |
| multi-assay-wide | kd_via_ic50 | II | 183 | 0.5912 / 0.8203 | 123 | 0.2772 / 0.8730 | 1.1113 |
| multi-assay-wide | kd_via_ec50 | I | 1175 | 0.5999 / 1.1873 | 1154 | 0.5830 / 1.1926 | 1.2101 |
| multi-assay-wide | kd_via_ec50 | II | 1529 | 0.5818 / 0.7259 | 1231 | 0.5336 / 0.7139 | 0.8433 |
| multi-assay-wide | half_life | I | 968 | 0.7404 / 0.8049 | 934 | 0.6995 / 0.8256 | 1.1847 |
| multi-assay-wide | half_life | II | 0 | — / — | 2 | 1.0000 / 1.5388 | 2.2367 |

## Runtime and selected checkpoint

All fits completed 6,000 updates. Requested H100!; actual NVIDIA H100 80GB HBM3.
Wall time excludes queue, image build, archive creation and download.
RSS is sampled at updates/end; CUDA peak is the allocator counter.

| Condition | Selected update | Wall seconds | Sampled RSS GiB | CUDA allocated peak GiB |
|---|---:|---:|---:|---:|
| ic50-exact-s42 | 5000 | 1316.6 | 4.96 | 0.84 |
| ic50-bounds-s42 | 6000 | 1379.9 | 4.97 | 0.81 |
| multi-assay-s42 | 6000 | 2086.3 | 6.05 | 0.67 |
| multi-assay-wide-s42 | 6000 | 2279.8 | 6.06 | 1.36 |
| ic50-exact-s43 | 6000 | 1287.8 | 4.98 | 0.78 |
| ic50-bounds-s43 | 5000 | 1306.7 | 4.99 | 0.79 |
| multi-assay-s43 | 5500 | 2017.4 | 6.06 | 0.70 |
| multi-assay-wide-s43 | 5500 | 2121.0 | 6.06 | 1.42 |
