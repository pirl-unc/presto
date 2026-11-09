## Seen fraction of held-out exact-BA test rows (identical rows in every family)

| class | tool | variant | rows | seen | fraction |
|---|---|---|---|---|---|
| I | mhcflurry | nested | 1313 | 721 | 0.549 |
| I | mhcflurry | pair | 1313 | 721 | 0.549 |
| I | mhcflurry | peptide | 1313 | 721 | 0.549 |
| I | netmhcpan41 | nested | 1313 | 564 | 0.430 |
| I | netmhcpan41 | pair | 1313 | 564 | 0.430 |
| I | netmhcpan41 | peptide | 1313 | 564 | 0.430 |
| II | netmhciipan43 | nested | 982 | 668 | 0.680 |
| II | netmhciipan43 | pair | 982 | 633 | 0.645 |
| II | netmhciipan43 | peptide | 982 | 660 | 0.672 |

## Seen fraction per study (pair variant, direct-s43_test, exact BA)

| class | tool | detector | study | n | seen | fraction |
|---|---|---|---|---|---|---|
| I | netmhcpan41 | fluorescence | pmid:17291638 | 68 | 68 | 1.000 |
| I | netmhcpan41 | fluorescence | pmid:34305923 | 157 | 0 | 0.000 |
| I | netmhcpan41 | fluorescence | reference:http://www.iedb.org/reference/1019353 | 384 | 384 | 1.000 |
| I | netmhcpan41 | radioactivity | pmid:17942551 | 112 | 112 | 1.000 |
| I | netmhcpan41 | radioactivity | pmid:42048449 | 592 | 0 | 0.000 |
| I | mhcflurry | fluorescence | pmid:17291638 | 68 | 68 | 1.000 |
| I | mhcflurry | fluorescence | pmid:34305923 | 157 | 157 | 1.000 |
| I | mhcflurry | fluorescence | reference:http://www.iedb.org/reference/1019353 | 384 | 384 | 1.000 |
| I | mhcflurry | radioactivity | pmid:17942551 | 112 | 112 | 1.000 |
| I | mhcflurry | radioactivity | pmid:42048449 | 592 | 0 | 0.000 |
| II | netmhciipan43 | fluorescence | pmid:11433380 | 144 | 116 | 0.806 |
| II | netmhciipan43 | fluorescence | pmid:24788397 | 194 | 187 | 0.964 |
| II | netmhciipan43 | fluorescence | pmid:34305923 | 296 | 0 | 0.000 |
| II | netmhciipan43 | fluorescence | pmid:40313951 | 21 | 3 | 0.143 |
| II | netmhciipan43 | fluorescence | reference:http://www.iedb.org/reference/1028237 | 149 | 149 | 1.000 |
| II | netmhciipan43 | radioactivity | pmid:17942551 | 178 | 178 | 1.000 |

## Tool-minus-Presto gap, direct-s42_test, variant=pair, own clean subset

Gap > 0 favours the tool for Spearman/Pearson/AUROC; for RMSE gap < 0 favours the tool. `full` = all PR #149 studies; `matched` = full restricted to the studies surviving in the clean cell; `clean` = rows the tool has not seen.

| class | detector | tool | metric | full gap (studies/rows) | matched gap | clean gap [CI] | clean studies/rows | tool / Presto on clean | status |
|---|---|---|---|---|---|---|---|---|---|
| I | fluorescence | mhcflurry | auroc | 0.246 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | pearson | 0.170 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | rmse | -0.521 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | spearman | 0.177 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | netmhcpan41 | auroc | 0.243 (3/607) | 0.115 | 0.115 [0.115, 0.115] | 1/157 | 0.685 / 0.569 | thin(<3 studies) |
| I | fluorescence | netmhcpan41 | pearson | 0.227 (3/607) | 0.137 | 0.137 [0.137, 0.137] | 1/157 | 0.391 / 0.255 | thin(<3 studies) |
| I | fluorescence | netmhcpan41 | rmse | -0.635 (3/607) | 0.542 | 0.542 [0.542, 0.542] | 1/157 | 1.622 / 1.080 | thin(<3 studies) |
| I | fluorescence | netmhcpan41 | spearman | 0.223 (3/607) | 0.148 | 0.148 [0.148, 0.148] | 1/157 | 0.383 / 0.234 | thin(<3 studies) |
| I | radioactivity | mhcflurry | auroc | 0.181 (2/692) | 0.194 | 0.194 [0.194, 0.194] | 1/592 | 0.880 / 0.686 | thin(<3 studies) |
| I | radioactivity | mhcflurry | pearson | 0.267 (2/692) | 0.382 | 0.382 [0.382, 0.382] | 1/592 | 0.746 / 0.364 | thin(<3 studies) |
| I | radioactivity | mhcflurry | rmse | -0.134 (2/692) | -0.219 | -0.219 [-0.219, -0.219] | 1/592 | 1.329 / 1.548 | thin(<3 studies) |
| I | radioactivity | mhcflurry | spearman | 0.256 (2/692) | 0.350 | 0.350 [0.350, 0.350] | 1/592 | 0.791 / 0.441 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | auroc | 0.258 (2/692) | 0.218 | 0.218 [0.218, 0.218] | 1/592 | 0.904 / 0.686 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | pearson | 0.346 (2/692) | 0.439 | 0.439 [0.439, 0.439] | 1/592 | 0.803 / 0.364 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | rmse | -0.363 (2/692) | -0.631 | -0.631 [-0.631, -0.631] | 1/592 | 0.917 / 1.548 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | spearman | 0.314 (2/692) | 0.381 | 0.381 [0.381, 0.381] | 1/592 | 0.822 / 0.441 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | auroc | 0.198 (5/804) | 0.189 | 0.061 [0.061, 0.061] | 1/296 | 0.726 / 0.666 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | pearson | 0.348 (5/804) | 0.326 | 0.200 [0.200, 0.200] | 1/296 | 0.452 / 0.252 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | rmse | -0.231 (5/804) | -0.138 | 0.209 [0.012, 0.405] | 2/324 | 1.158 / 0.950 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | spearman | 0.308 (5/804) | 0.267 | 0.153 [0.153, 0.153] | 1/296 | 0.388 / 0.234 | thin(<3 studies) |
| II | radioactivity | netmhciipan43 | auroc | 0.275 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | pearson | 0.391 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | rmse | -0.359 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | spearman | 0.381 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |

## Tool-minus-Presto gap, direct-s42_test, variant=peptide, own clean subset

Gap > 0 favours the tool for Spearman/Pearson/AUROC; for RMSE gap < 0 favours the tool. `full` = all PR #149 studies; `matched` = full restricted to the studies surviving in the clean cell; `clean` = rows the tool has not seen.

| class | detector | tool | metric | full gap (studies/rows) | matched gap | clean gap [CI] | clean studies/rows | tool / Presto on clean | status |
|---|---|---|---|---|---|---|---|---|---|
| I | fluorescence | mhcflurry | auroc | 0.246 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | pearson | 0.170 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | rmse | -0.521 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | spearman | 0.177 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | netmhcpan41 | auroc | 0.243 (3/607) | 0.115 | 0.115 [0.115, 0.115] | 1/157 | 0.685 / 0.569 | thin(<3 studies) |
| I | fluorescence | netmhcpan41 | pearson | 0.227 (3/607) | 0.137 | 0.137 [0.137, 0.137] | 1/157 | 0.391 / 0.255 | thin(<3 studies) |
| I | fluorescence | netmhcpan41 | rmse | -0.635 (3/607) | 0.542 | 0.542 [0.542, 0.542] | 1/157 | 1.622 / 1.080 | thin(<3 studies) |
| I | fluorescence | netmhcpan41 | spearman | 0.223 (3/607) | 0.148 | 0.148 [0.148, 0.148] | 1/157 | 0.383 / 0.234 | thin(<3 studies) |
| I | radioactivity | mhcflurry | auroc | 0.181 (2/692) | 0.194 | 0.194 [0.194, 0.194] | 1/592 | 0.880 / 0.686 | thin(<3 studies) |
| I | radioactivity | mhcflurry | pearson | 0.267 (2/692) | 0.382 | 0.382 [0.382, 0.382] | 1/592 | 0.746 / 0.364 | thin(<3 studies) |
| I | radioactivity | mhcflurry | rmse | -0.134 (2/692) | -0.219 | -0.219 [-0.219, -0.219] | 1/592 | 1.329 / 1.548 | thin(<3 studies) |
| I | radioactivity | mhcflurry | spearman | 0.256 (2/692) | 0.350 | 0.350 [0.350, 0.350] | 1/592 | 0.791 / 0.441 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | auroc | 0.258 (2/692) | 0.218 | 0.218 [0.218, 0.218] | 1/592 | 0.904 / 0.686 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | pearson | 0.346 (2/692) | 0.439 | 0.439 [0.439, 0.439] | 1/592 | 0.803 / 0.364 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | rmse | -0.363 (2/692) | -0.631 | -0.631 [-0.631, -0.631] | 1/592 | 0.917 / 1.548 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | spearman | 0.314 (2/692) | 0.381 | 0.381 [0.381, 0.381] | 1/592 | 0.822 / 0.441 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | auroc | 0.198 (5/804) | 0.061 | 0.061 [0.061, 0.061] | 1/296 | 0.726 / 0.666 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | pearson | 0.348 (5/804) | 0.200 | 0.200 [0.200, 0.200] | 1/296 | 0.452 / 0.252 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | rmse | -0.231 (5/804) | 0.012 | 0.012 [0.012, 0.012] | 1/296 | 1.360 / 1.348 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | spearman | 0.308 (5/804) | 0.153 | 0.153 [0.153, 0.153] | 1/296 | 0.388 / 0.234 | thin(<3 studies) |
| II | radioactivity | netmhciipan43 | auroc | 0.275 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | pearson | 0.391 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | rmse | -0.359 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | spearman | 0.381 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |

## Tool-minus-Presto gap, direct-s42_test, variant=nested, own clean subset

Gap > 0 favours the tool for Spearman/Pearson/AUROC; for RMSE gap < 0 favours the tool. `full` = all PR #149 studies; `matched` = full restricted to the studies surviving in the clean cell; `clean` = rows the tool has not seen.

| class | detector | tool | metric | full gap (studies/rows) | matched gap | clean gap [CI] | clean studies/rows | tool / Presto on clean | status |
|---|---|---|---|---|---|---|---|---|---|
| I | fluorescence | mhcflurry | auroc | 0.246 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | pearson | 0.170 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | rmse | -0.521 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | spearman | 0.177 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | netmhcpan41 | auroc | 0.243 (3/607) | 0.115 | 0.115 [0.115, 0.115] | 1/157 | 0.685 / 0.569 | thin(<3 studies) |
| I | fluorescence | netmhcpan41 | pearson | 0.227 (3/607) | 0.137 | 0.137 [0.137, 0.137] | 1/157 | 0.391 / 0.255 | thin(<3 studies) |
| I | fluorescence | netmhcpan41 | rmse | -0.635 (3/607) | 0.542 | 0.542 [0.542, 0.542] | 1/157 | 1.622 / 1.080 | thin(<3 studies) |
| I | fluorescence | netmhcpan41 | spearman | 0.223 (3/607) | 0.148 | 0.148 [0.148, 0.148] | 1/157 | 0.383 / 0.234 | thin(<3 studies) |
| I | radioactivity | mhcflurry | auroc | 0.181 (2/692) | 0.194 | 0.194 [0.194, 0.194] | 1/592 | 0.880 / 0.686 | thin(<3 studies) |
| I | radioactivity | mhcflurry | pearson | 0.267 (2/692) | 0.382 | 0.382 [0.382, 0.382] | 1/592 | 0.746 / 0.364 | thin(<3 studies) |
| I | radioactivity | mhcflurry | rmse | -0.134 (2/692) | -0.219 | -0.219 [-0.219, -0.219] | 1/592 | 1.329 / 1.548 | thin(<3 studies) |
| I | radioactivity | mhcflurry | spearman | 0.256 (2/692) | 0.350 | 0.350 [0.350, 0.350] | 1/592 | 0.791 / 0.441 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | auroc | 0.258 (2/692) | 0.218 | 0.218 [0.218, 0.218] | 1/592 | 0.904 / 0.686 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | pearson | 0.346 (2/692) | 0.439 | 0.439 [0.439, 0.439] | 1/592 | 0.803 / 0.364 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | rmse | -0.363 (2/692) | -0.631 | -0.631 [-0.631, -0.631] | 1/592 | 0.917 / 1.548 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | spearman | 0.314 (2/692) | 0.381 | 0.381 [0.381, 0.381] | 1/592 | 0.822 / 0.441 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | auroc | 0.198 (5/804) | 0.061 | 0.061 [0.061, 0.061] | 1/296 | 0.726 / 0.666 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | pearson | 0.348 (5/804) | 0.200 | 0.200 [0.200, 0.200] | 1/296 | 0.452 / 0.252 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | rmse | -0.231 (5/804) | 0.012 | 0.012 [0.012, 0.012] | 1/296 | 1.360 / 1.348 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | spearman | 0.308 (5/804) | 0.153 | 0.153 [0.153, 0.153] | 1/296 | 0.388 / 0.234 | thin(<3 studies) |
| II | radioactivity | netmhciipan43 | auroc | 0.275 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | pearson | 0.391 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | rmse | -0.359 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | spearman | 0.381 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |

## Tool-minus-Presto gap, direct-s43_test, variant=pair, own clean subset

Gap > 0 favours the tool for Spearman/Pearson/AUROC; for RMSE gap < 0 favours the tool. `full` = all PR #149 studies; `matched` = full restricted to the studies surviving in the clean cell; `clean` = rows the tool has not seen.

| class | detector | tool | metric | full gap (studies/rows) | matched gap | clean gap [CI] | clean studies/rows | tool / Presto on clean | status |
|---|---|---|---|---|---|---|---|---|---|
| I | fluorescence | mhcflurry | auroc | 0.174 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | pearson | 0.131 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | rmse | -0.440 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | spearman | 0.146 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | netmhcpan41 | auroc | 0.171 (3/607) | 0.100 | 0.100 [0.100, 0.100] | 1/157 | 0.685 / 0.584 | thin(<3 studies) |
| I | fluorescence | netmhcpan41 | pearson | 0.188 (3/607) | 0.080 | 0.080 [0.080, 0.080] | 1/157 | 0.391 / 0.311 | thin(<3 studies) |
| I | fluorescence | netmhcpan41 | rmse | -0.554 (3/607) | 0.463 | 0.463 [0.463, 0.463] | 1/157 | 1.622 / 1.159 | thin(<3 studies) |
| I | fluorescence | netmhcpan41 | spearman | 0.192 (3/607) | 0.068 | 0.068 [0.068, 0.068] | 1/157 | 0.383 / 0.315 | thin(<3 studies) |
| I | radioactivity | mhcflurry | auroc | 0.191 (2/692) | 0.326 | 0.326 [0.326, 0.326] | 1/592 | 0.880 / 0.554 | thin(<3 studies) |
| I | radioactivity | mhcflurry | pearson | 0.261 (2/692) | 0.493 | 0.493 [0.493, 0.493] | 1/592 | 0.746 / 0.253 | thin(<3 studies) |
| I | radioactivity | mhcflurry | rmse | -0.066 (2/692) | -0.098 | -0.098 [-0.098, -0.098] | 1/592 | 1.329 / 1.427 | thin(<3 studies) |
| I | radioactivity | mhcflurry | spearman | 0.279 (2/692) | 0.488 | 0.488 [0.488, 0.488] | 1/592 | 0.791 / 0.303 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | auroc | 0.267 (2/692) | 0.350 | 0.350 [0.350, 0.350] | 1/592 | 0.904 / 0.554 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | pearson | 0.340 (2/692) | 0.550 | 0.550 [0.550, 0.550] | 1/592 | 0.803 / 0.253 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | rmse | -0.294 (2/692) | -0.509 | -0.509 [-0.509, -0.509] | 1/592 | 0.917 / 1.427 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | spearman | 0.337 (2/692) | 0.519 | 0.519 [0.519, 0.519] | 1/592 | 0.822 / 0.303 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | auroc | 0.219 (5/804) | 0.222 | 0.086 [0.086, 0.086] | 1/296 | 0.726 / 0.640 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | pearson | 0.353 (5/804) | 0.327 | 0.219 [0.219, 0.219] | 1/296 | 0.452 / 0.233 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | rmse | -0.248 (5/804) | -0.075 | 0.287 [0.122, 0.452] | 2/324 | 1.158 / 0.871 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | spearman | 0.329 (5/804) | 0.274 | 0.145 [0.145, 0.145] | 1/296 | 0.388 / 0.243 | thin(<3 studies) |
| II | radioactivity | netmhciipan43 | auroc | 0.277 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | pearson | 0.253 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | rmse | -0.255 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | spearman | 0.267 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |

## Tool-minus-Presto gap, direct-s43_test, variant=peptide, own clean subset

Gap > 0 favours the tool for Spearman/Pearson/AUROC; for RMSE gap < 0 favours the tool. `full` = all PR #149 studies; `matched` = full restricted to the studies surviving in the clean cell; `clean` = rows the tool has not seen.

| class | detector | tool | metric | full gap (studies/rows) | matched gap | clean gap [CI] | clean studies/rows | tool / Presto on clean | status |
|---|---|---|---|---|---|---|---|---|---|
| I | fluorescence | mhcflurry | auroc | 0.174 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | pearson | 0.131 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | rmse | -0.440 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | spearman | 0.146 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | netmhcpan41 | auroc | 0.171 (3/607) | 0.100 | 0.100 [0.100, 0.100] | 1/157 | 0.685 / 0.584 | thin(<3 studies) |
| I | fluorescence | netmhcpan41 | pearson | 0.188 (3/607) | 0.080 | 0.080 [0.080, 0.080] | 1/157 | 0.391 / 0.311 | thin(<3 studies) |
| I | fluorescence | netmhcpan41 | rmse | -0.554 (3/607) | 0.463 | 0.463 [0.463, 0.463] | 1/157 | 1.622 / 1.159 | thin(<3 studies) |
| I | fluorescence | netmhcpan41 | spearman | 0.192 (3/607) | 0.068 | 0.068 [0.068, 0.068] | 1/157 | 0.383 / 0.315 | thin(<3 studies) |
| I | radioactivity | mhcflurry | auroc | 0.191 (2/692) | 0.326 | 0.326 [0.326, 0.326] | 1/592 | 0.880 / 0.554 | thin(<3 studies) |
| I | radioactivity | mhcflurry | pearson | 0.261 (2/692) | 0.493 | 0.493 [0.493, 0.493] | 1/592 | 0.746 / 0.253 | thin(<3 studies) |
| I | radioactivity | mhcflurry | rmse | -0.066 (2/692) | -0.098 | -0.098 [-0.098, -0.098] | 1/592 | 1.329 / 1.427 | thin(<3 studies) |
| I | radioactivity | mhcflurry | spearman | 0.279 (2/692) | 0.488 | 0.488 [0.488, 0.488] | 1/592 | 0.791 / 0.303 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | auroc | 0.267 (2/692) | 0.350 | 0.350 [0.350, 0.350] | 1/592 | 0.904 / 0.554 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | pearson | 0.340 (2/692) | 0.550 | 0.550 [0.550, 0.550] | 1/592 | 0.803 / 0.253 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | rmse | -0.294 (2/692) | -0.509 | -0.509 [-0.509, -0.509] | 1/592 | 0.917 / 1.427 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | spearman | 0.337 (2/692) | 0.519 | 0.519 [0.519, 0.519] | 1/592 | 0.822 / 0.303 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | auroc | 0.219 (5/804) | 0.086 | 0.086 [0.086, 0.086] | 1/296 | 0.726 / 0.640 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | pearson | 0.353 (5/804) | 0.219 | 0.219 [0.219, 0.219] | 1/296 | 0.452 / 0.233 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | rmse | -0.248 (5/804) | 0.122 | 0.122 [0.122, 0.122] | 1/296 | 1.360 / 1.238 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | spearman | 0.329 (5/804) | 0.145 | 0.145 [0.145, 0.145] | 1/296 | 0.388 / 0.243 | thin(<3 studies) |
| II | radioactivity | netmhciipan43 | auroc | 0.277 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | pearson | 0.253 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | rmse | -0.255 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | spearman | 0.267 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |

## Tool-minus-Presto gap, direct-s43_test, variant=nested, own clean subset

Gap > 0 favours the tool for Spearman/Pearson/AUROC; for RMSE gap < 0 favours the tool. `full` = all PR #149 studies; `matched` = full restricted to the studies surviving in the clean cell; `clean` = rows the tool has not seen.

| class | detector | tool | metric | full gap (studies/rows) | matched gap | clean gap [CI] | clean studies/rows | tool / Presto on clean | status |
|---|---|---|---|---|---|---|---|---|---|
| I | fluorescence | mhcflurry | auroc | 0.174 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | pearson | 0.131 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | rmse | -0.440 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | spearman | 0.146 (3/607) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | netmhcpan41 | auroc | 0.171 (3/607) | 0.100 | 0.100 [0.100, 0.100] | 1/157 | 0.685 / 0.584 | thin(<3 studies) |
| I | fluorescence | netmhcpan41 | pearson | 0.188 (3/607) | 0.080 | 0.080 [0.080, 0.080] | 1/157 | 0.391 / 0.311 | thin(<3 studies) |
| I | fluorescence | netmhcpan41 | rmse | -0.554 (3/607) | 0.463 | 0.463 [0.463, 0.463] | 1/157 | 1.622 / 1.159 | thin(<3 studies) |
| I | fluorescence | netmhcpan41 | spearman | 0.192 (3/607) | 0.068 | 0.068 [0.068, 0.068] | 1/157 | 0.383 / 0.315 | thin(<3 studies) |
| I | radioactivity | mhcflurry | auroc | 0.191 (2/692) | 0.326 | 0.326 [0.326, 0.326] | 1/592 | 0.880 / 0.554 | thin(<3 studies) |
| I | radioactivity | mhcflurry | pearson | 0.261 (2/692) | 0.493 | 0.493 [0.493, 0.493] | 1/592 | 0.746 / 0.253 | thin(<3 studies) |
| I | radioactivity | mhcflurry | rmse | -0.066 (2/692) | -0.098 | -0.098 [-0.098, -0.098] | 1/592 | 1.329 / 1.427 | thin(<3 studies) |
| I | radioactivity | mhcflurry | spearman | 0.279 (2/692) | 0.488 | 0.488 [0.488, 0.488] | 1/592 | 0.791 / 0.303 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | auroc | 0.267 (2/692) | 0.350 | 0.350 [0.350, 0.350] | 1/592 | 0.904 / 0.554 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | pearson | 0.340 (2/692) | 0.550 | 0.550 [0.550, 0.550] | 1/592 | 0.803 / 0.253 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | rmse | -0.294 (2/692) | -0.509 | -0.509 [-0.509, -0.509] | 1/592 | 0.917 / 1.427 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | spearman | 0.337 (2/692) | 0.519 | 0.519 [0.519, 0.519] | 1/592 | 0.822 / 0.303 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | auroc | 0.219 (5/804) | 0.086 | 0.086 [0.086, 0.086] | 1/296 | 0.726 / 0.640 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | pearson | 0.353 (5/804) | 0.219 | 0.219 [0.219, 0.219] | 1/296 | 0.452 / 0.233 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | rmse | -0.248 (5/804) | 0.122 | 0.122 [0.122, 0.122] | 1/296 | 1.360 / 1.238 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | spearman | 0.329 (5/804) | 0.145 | 0.145 [0.145, 0.145] | 1/296 | 0.388 / 0.243 | thin(<3 studies) |
| II | radioactivity | netmhciipan43 | auroc | 0.277 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | pearson | 0.253 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | rmse | -0.255 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | spearman | 0.267 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |

## Tool-minus-Presto gap, curated-multi-s43_test, variant=pair, own clean subset

Gap > 0 favours the tool for Spearman/Pearson/AUROC; for RMSE gap < 0 favours the tool. `full` = all PR #149 studies; `matched` = full restricted to the studies surviving in the clean cell; `clean` = rows the tool has not seen.

| class | detector | tool | metric | full gap (studies/rows) | matched gap | clean gap [CI] | clean studies/rows | tool / Presto on clean | status |
|---|---|---|---|---|---|---|---|---|---|
| I | fluorescence | mhcflurry | auroc | -0.042 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | pearson | 0.137 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | rmse | -0.101 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | spearman | 0.184 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | netmhcpan41 | auroc | -0.011 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | netmhcpan41 | pearson | 0.260 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | netmhcpan41 | rmse | -0.222 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | netmhcpan41 | spearman | 0.302 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | radioactivity | mhcflurry | auroc | -0.034 (2/692) | 0.012 | 0.012 [0.012, 0.012] | 1/592 | 0.880 / 0.868 | thin(<3 studies) |
| I | radioactivity | mhcflurry | pearson | 0.040 (2/692) | 0.021 | 0.021 [0.021, 0.021] | 1/592 | 0.746 / 0.724 | thin(<3 studies) |
| I | radioactivity | mhcflurry | rmse | 0.084 (2/692) | 0.241 | 0.241 [0.241, 0.241] | 1/592 | 1.329 / 1.088 | thin(<3 studies) |
| I | radioactivity | mhcflurry | spearman | 0.079 (2/692) | 0.091 | 0.091 [0.091, 0.091] | 1/592 | 0.791 / 0.700 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | auroc | 0.043 (2/692) | 0.036 | 0.036 [0.036, 0.036] | 1/592 | 0.904 / 0.868 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | pearson | 0.120 (2/692) | 0.078 | 0.078 [0.078, 0.078] | 1/592 | 0.803 / 0.724 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | rmse | -0.144 (2/692) | -0.170 | -0.170 [-0.170, -0.170] | 1/592 | 0.917 / 1.088 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | spearman | 0.137 (2/692) | 0.122 | 0.122 [0.122, 0.122] | 1/592 | 0.822 / 0.700 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | auroc | 0.141 (3/314) | 0.253 | n/a | 0/0 |  /  | no |
| II | fluorescence | netmhciipan43 | pearson | 0.219 (3/314) | 0.226 | n/a | 0/0 |  /  | no |
| II | fluorescence | netmhciipan43 | rmse | -0.012 (3/314) | -0.139 | 0.368 [0.368, 0.368] | 1/28 | 0.956 / 0.588 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | spearman | 0.213 (3/314) | 0.242 | n/a | 0/0 |  /  | no |
| II | radioactivity | netmhciipan43 | auroc | 0.157 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | pearson | 0.134 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | rmse | -0.452 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | spearman | 0.151 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |

## Tool-minus-Presto gap, curated-multi-s43_test, variant=peptide, own clean subset

Gap > 0 favours the tool for Spearman/Pearson/AUROC; for RMSE gap < 0 favours the tool. `full` = all PR #149 studies; `matched` = full restricted to the studies surviving in the clean cell; `clean` = rows the tool has not seen.

| class | detector | tool | metric | full gap (studies/rows) | matched gap | clean gap [CI] | clean studies/rows | tool / Presto on clean | status |
|---|---|---|---|---|---|---|---|---|---|
| I | fluorescence | mhcflurry | auroc | -0.042 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | pearson | 0.137 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | rmse | -0.101 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | spearman | 0.184 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | netmhcpan41 | auroc | -0.011 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | netmhcpan41 | pearson | 0.260 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | netmhcpan41 | rmse | -0.222 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | netmhcpan41 | spearman | 0.302 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | radioactivity | mhcflurry | auroc | -0.034 (2/692) | 0.012 | 0.012 [0.012, 0.012] | 1/592 | 0.880 / 0.868 | thin(<3 studies) |
| I | radioactivity | mhcflurry | pearson | 0.040 (2/692) | 0.021 | 0.021 [0.021, 0.021] | 1/592 | 0.746 / 0.724 | thin(<3 studies) |
| I | radioactivity | mhcflurry | rmse | 0.084 (2/692) | 0.241 | 0.241 [0.241, 0.241] | 1/592 | 1.329 / 1.088 | thin(<3 studies) |
| I | radioactivity | mhcflurry | spearman | 0.079 (2/692) | 0.091 | 0.091 [0.091, 0.091] | 1/592 | 0.791 / 0.700 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | auroc | 0.043 (2/692) | 0.036 | 0.036 [0.036, 0.036] | 1/592 | 0.904 / 0.868 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | pearson | 0.120 (2/692) | 0.078 | 0.078 [0.078, 0.078] | 1/592 | 0.803 / 0.724 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | rmse | -0.144 (2/692) | -0.170 | -0.170 [-0.170, -0.170] | 1/592 | 0.917 / 1.088 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | spearman | 0.137 (2/692) | 0.122 | 0.122 [0.122, 0.122] | 1/592 | 0.822 / 0.700 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | auroc | 0.141 (3/314) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | fluorescence | netmhciipan43 | pearson | 0.219 (3/314) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | fluorescence | netmhciipan43 | rmse | -0.012 (3/314) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | fluorescence | netmhciipan43 | spearman | 0.213 (3/314) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | auroc | 0.157 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | pearson | 0.134 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | rmse | -0.452 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | spearman | 0.151 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |

## Tool-minus-Presto gap, curated-multi-s43_test, variant=nested, own clean subset

Gap > 0 favours the tool for Spearman/Pearson/AUROC; for RMSE gap < 0 favours the tool. `full` = all PR #149 studies; `matched` = full restricted to the studies surviving in the clean cell; `clean` = rows the tool has not seen.

| class | detector | tool | metric | full gap (studies/rows) | matched gap | clean gap [CI] | clean studies/rows | tool / Presto on clean | status |
|---|---|---|---|---|---|---|---|---|---|
| I | fluorescence | mhcflurry | auroc | -0.042 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | pearson | 0.137 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | rmse | -0.101 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | mhcflurry | spearman | 0.184 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | netmhcpan41 | auroc | -0.011 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | netmhcpan41 | pearson | 0.260 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | netmhcpan41 | rmse | -0.222 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | fluorescence | netmhcpan41 | spearman | 0.302 (2/450) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| I | radioactivity | mhcflurry | auroc | -0.034 (2/692) | 0.012 | 0.012 [0.012, 0.012] | 1/592 | 0.880 / 0.868 | thin(<3 studies) |
| I | radioactivity | mhcflurry | pearson | 0.040 (2/692) | 0.021 | 0.021 [0.021, 0.021] | 1/592 | 0.746 / 0.724 | thin(<3 studies) |
| I | radioactivity | mhcflurry | rmse | 0.084 (2/692) | 0.241 | 0.241 [0.241, 0.241] | 1/592 | 1.329 / 1.088 | thin(<3 studies) |
| I | radioactivity | mhcflurry | spearman | 0.079 (2/692) | 0.091 | 0.091 [0.091, 0.091] | 1/592 | 0.791 / 0.700 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | auroc | 0.043 (2/692) | 0.036 | 0.036 [0.036, 0.036] | 1/592 | 0.904 / 0.868 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | pearson | 0.120 (2/692) | 0.078 | 0.078 [0.078, 0.078] | 1/592 | 0.803 / 0.724 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | rmse | -0.144 (2/692) | -0.170 | -0.170 [-0.170, -0.170] | 1/592 | 0.917 / 1.088 | thin(<3 studies) |
| I | radioactivity | netmhcpan41 | spearman | 0.137 (2/692) | 0.122 | 0.122 [0.122, 0.122] | 1/592 | 0.822 / 0.700 | thin(<3 studies) |
| II | fluorescence | netmhciipan43 | auroc | 0.141 (3/314) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | fluorescence | netmhciipan43 | pearson | 0.219 (3/314) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | fluorescence | netmhciipan43 | rmse | -0.012 (3/314) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | fluorescence | netmhciipan43 | spearman | 0.213 (3/314) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | auroc | 0.157 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | pearson | 0.134 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | rmse | -0.452 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
| II | radioactivity | netmhciipan43 | spearman | 0.151 (1/178) |  | n/a | 0/0 |  /  | no(clean n<20 everywhere) |
