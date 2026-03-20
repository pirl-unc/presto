# 50k vs 100k Summary
This note compares cap changes within the same target encoding for the broad target-space bakeoff.
Important: this experiment saved aggregate/probe metrics only. It did **not** save per-example validation predictions, so exact-IC50 rank correlation and `<=500 nM` accuracy cannot be reconstructed from these artifacts.
## A03
| encoding | 50k best val | 100k best val | 50k SLL ratio | 100k SLL ratio | 50k FLR ratio | 100k FLR ratio | 50k NFL ratio | 100k NFL ratio | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `log10` | 0.9503 | 0.9847 | 2746.2 | 2261.5 | 466.8 | 193.9 | 0.9 | 7.6 | 50k better SLL, 100k better NFL, 50k better val |
| `mhcflurry` | 0.0410 | 0.0371 | 486.7 | 784.0 | 1096.8 | 258.5 | 3.7 | 7.4 | 100k better SLL, 100k better NFL, 100k better val |

## A07
| encoding | 50k best val | 100k best val | 50k SLL ratio | 100k SLL ratio | 50k FLR ratio | 100k FLR ratio | 50k NFL ratio | 100k NFL ratio | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `log10` | 0.7580 | 0.8137 | 169.7 | 70.7 | 41.1 | 46.1 | 99.9 | 92.3 | 50k better SLL, 50k better NFL, 50k better val |
| `mhcflurry` | 0.0326 | 0.0359 | 382.3 | 491.8 | 175.1 | 201.8 | 51.9 | 119.9 | 100k better SLL, 100k better NFL, 50k better val |

