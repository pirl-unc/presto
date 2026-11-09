### direct-s42_test (held-out test)

| class | detector | method | studies | rows | Spearman [95% CI] | Fisher-z rho | Pearson | RMSE log10 | AUROC<=500nM | bal.acc | Spearman gap vs Presto [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I | fluorescence | presto | 3 | 607 | 0.371 [0.186, 0.694] | 0.265 | 0.367 | 1.765 | 0.511 | 0.550 |  |
| I | fluorescence | netmhcpan41 | 3 | 607 | 0.595 [0.383, 0.724] | 0.620 | 0.594 | 1.131 | 0.754 | 0.592 | +0.223 [+0.030, +0.491] |
| I | fluorescence | mhcflurry | 3 | 607 | 0.548 [0.480, 0.652] | 0.521 | 0.537 | 1.244 | 0.757 | 0.575 | +0.177 [-0.042, +0.327] |
| I | fluorescence | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.429 | 0.500 | 0.500 |  |
| I | radioactivity | presto | 2 | 692 | 0.324 [0.207, 0.441] | 0.410 | 0.279 | 1.285 | 0.601 | 0.583 |  |
| I | radioactivity | netmhcpan41 | 2 | 692 | 0.638 [0.455, 0.822] | 0.788 | 0.625 | 0.922 | 0.859 | 0.772 | +0.314 [+0.248, +0.381] |
| I | radioactivity | mhcflurry | 2 | 692 | 0.580 [0.369, 0.791] | 0.752 | 0.546 | 1.151 | 0.782 | 0.640 | +0.256 [+0.162, +0.350] |
| I | radioactivity | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.548 | 0.500 | 0.500 |  |
| II | fluorescence | presto | 5 | 804 | 0.389 [0.245, 0.541] | 0.312 | 0.370 | 1.173 | 0.674 | 0.532 |  |
| II | fluorescence | netmhciipan43 | 5 | 804 | 0.697 [0.531, 0.817] | 0.693 | 0.718 | 0.942 | 0.871 | 0.727 | +0.308 [+0.147, +0.469] |
| II | fluorescence | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.358 | 0.500 | 0.500 |  |
| II | radioactivity | presto | 1 | 178 | 0.053 [0.053, 0.053] | 0.108 | 0.031 | 0.972 | 0.586 | 0.377 |  |
| II | radioactivity | netmhciipan43 | 1 | 178 | 0.434 [0.434, 0.434] | 0.557 | 0.422 | 0.614 | 0.861 | 0.570 | +0.381 [+0.381, +0.381] |
| II | radioactivity | train_mean | 0 | 0 | undefined (constant) | nan | nan | 0.947 | 0.500 | 0.250 |  |

### direct-s42_validation (DEVELOPMENTAL (validation))

| class | detector | method | studies | rows | Spearman [95% CI] | Fisher-z rho | Pearson | RMSE log10 | AUROC<=500nM | bal.acc | Spearman gap vs Presto [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I | fluorescence | presto | 3 | 285 | 0.402 [0.250, 0.531] | 0.372 | 0.425 | 1.571 | 0.788 | 0.430 |  |
| I | fluorescence | netmhcpan41 | 3 | 285 | 0.689 [0.618, 0.829] | 0.749 | 0.706 | 0.988 | 0.910 | 0.701 | +0.287 [+0.086, +0.406] |
| I | fluorescence | mhcflurry | 3 | 285 | 0.671 [0.574, 0.742] | 0.659 | 0.675 | 0.986 | 0.913 | 0.686 | +0.269 [+0.210, +0.324] |
| I | fluorescence | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.256 | 0.500 | 0.500 |  |
| I | radioactivity | presto | 1 | 208 | 0.312 [0.312, 0.312] | 0.371 | 0.353 | 1.211 | nan | 0.386 |  |
| I | radioactivity | netmhcpan41 | 1 | 208 | 0.628 [0.628, 0.628] | 0.680 | 0.598 | 0.810 | nan | 0.432 | +0.317 [+0.317, +0.317] |
| I | radioactivity | mhcflurry | 1 | 208 | 0.625 [0.625, 0.625] | 0.667 | 0.613 | 0.923 | nan | 0.402 | +0.314 [+0.314, +0.314] |
| I | radioactivity | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.615 | nan | 0.000 |  |
| II | fluorescence | presto | 2 | 393 | 0.667 [0.490, 0.845] | 0.541 | 0.658 | 0.591 | 0.808 | 0.578 |  |
| II | fluorescence | netmhciipan43 | 2 | 393 | 0.850 [0.842, 0.859] | 0.857 | 0.868 | 0.416 | 0.932 | 0.802 | +0.183 [-0.003, +0.369] |
| II | fluorescence | train_mean | 0 | 0 | undefined (constant) | nan | nan | 0.852 | 0.500 | 0.500 |  |
| II | radioactivity | presto | 1 | 1375 | 0.262 [0.262, 0.262] | 0.322 | 0.246 | 1.115 | 0.685 | 0.505 |  |
| II | radioactivity | netmhciipan43 | 1 | 1375 | 0.460 [0.460, 0.460] | 0.701 | 0.468 | 0.635 | 0.900 | 0.665 | +0.198 [+0.198, +0.198] |
| II | radioactivity | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.240 | 0.500 | 0.250 |  |

### direct-s43_test (held-out test)

| class | detector | method | studies | rows | Spearman [95% CI] | Fisher-z rho | Pearson | RMSE log10 | AUROC<=500nM | bal.acc | Spearman gap vs Presto [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I | fluorescence | presto | 3 | 607 | 0.402 [0.142, 0.751] | 0.271 | 0.406 | 1.684 | 0.583 | 0.587 |  |
| I | fluorescence | netmhcpan41 | 3 | 607 | 0.595 [0.383, 0.724] | 0.620 | 0.594 | 1.131 | 0.754 | 0.592 | +0.192 [-0.027, +0.535] |
| I | fluorescence | mhcflurry | 3 | 607 | 0.548 [0.480, 0.652] | 0.521 | 0.537 | 1.244 | 0.757 | 0.575 | +0.146 [-0.099, +0.371] |
| I | fluorescence | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.429 | 0.500 | 0.500 |  |
| I | radioactivity | presto | 2 | 692 | 0.301 [0.299, 0.303] | 0.303 | 0.285 | 1.216 | 0.592 | 0.552 |  |
| I | radioactivity | netmhcpan41 | 2 | 692 | 0.638 [0.455, 0.822] | 0.788 | 0.625 | 0.922 | 0.859 | 0.772 | +0.337 [+0.156, +0.519] |
| I | radioactivity | mhcflurry | 2 | 692 | 0.580 [0.369, 0.791] | 0.752 | 0.546 | 1.151 | 0.782 | 0.640 | +0.279 [+0.070, +0.488] |
| I | radioactivity | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.548 | 0.500 | 0.500 |  |
| II | fluorescence | presto | 5 | 804 | 0.368 [0.233, 0.538] | 0.287 | 0.365 | 1.190 | 0.653 | 0.575 |  |
| II | fluorescence | netmhciipan43 | 5 | 804 | 0.697 [0.531, 0.817] | 0.693 | 0.718 | 0.942 | 0.871 | 0.727 | +0.329 [+0.141, +0.507] |
| II | fluorescence | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.358 | 0.500 | 0.500 |  |
| II | radioactivity | presto | 1 | 178 | 0.167 [0.167, 0.167] | 0.192 | 0.169 | 0.869 | 0.584 | 0.353 |  |
| II | radioactivity | netmhciipan43 | 1 | 178 | 0.434 [0.434, 0.434] | 0.557 | 0.422 | 0.614 | 0.861 | 0.570 | +0.267 [+0.267, +0.267] |
| II | radioactivity | train_mean | 0 | 0 | undefined (constant) | nan | nan | 0.947 | 0.500 | 0.250 |  |

### direct-s43_validation (DEVELOPMENTAL (validation))

| class | detector | method | studies | rows | Spearman [95% CI] | Fisher-z rho | Pearson | RMSE log10 | AUROC<=500nM | bal.acc | Spearman gap vs Presto [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I | fluorescence | presto | 3 | 285 | 0.396 [0.187, 0.583] | 0.353 | 0.401 | 1.701 | 0.751 | 0.441 |  |
| I | fluorescence | netmhcpan41 | 3 | 285 | 0.689 [0.618, 0.829] | 0.749 | 0.706 | 0.988 | 0.910 | 0.701 | +0.293 [+0.034, +0.432] |
| I | fluorescence | mhcflurry | 3 | 285 | 0.671 [0.574, 0.742] | 0.659 | 0.675 | 0.986 | 0.913 | 0.686 | +0.275 [+0.158, +0.387] |
| I | fluorescence | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.256 | 0.500 | 0.500 |  |
| I | radioactivity | presto | 1 | 208 | 0.400 [0.400, 0.400] | 0.441 | 0.423 | 1.087 | nan | 0.399 |  |
| I | radioactivity | netmhcpan41 | 1 | 208 | 0.628 [0.628, 0.628] | 0.680 | 0.598 | 0.810 | nan | 0.432 | +0.229 [+0.229, +0.229] |
| I | radioactivity | mhcflurry | 1 | 208 | 0.625 [0.625, 0.625] | 0.667 | 0.613 | 0.923 | nan | 0.402 | +0.226 [+0.226, +0.226] |
| I | radioactivity | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.615 | nan | 0.000 |  |
| II | fluorescence | presto | 2 | 393 | 0.614 [0.411, 0.817] | 0.467 | 0.615 | 0.523 | 0.775 | 0.666 |  |
| II | fluorescence | netmhciipan43 | 2 | 393 | 0.850 [0.842, 0.859] | 0.857 | 0.868 | 0.416 | 0.932 | 0.802 | +0.236 [+0.025, +0.448] |
| II | fluorescence | train_mean | 0 | 0 | undefined (constant) | nan | nan | 0.852 | 0.500 | 0.500 |  |
| II | radioactivity | presto | 1 | 1375 | 0.316 [0.316, 0.316] | 0.374 | 0.299 | 1.085 | 0.703 | 0.500 |  |
| II | radioactivity | netmhciipan43 | 1 | 1375 | 0.460 [0.460, 0.460] | 0.701 | 0.468 | 0.635 | 0.900 | 0.665 | +0.144 [+0.144, +0.144] |
| II | radioactivity | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.240 | 0.500 | 0.250 |  |

### curated-multi-s43_test (held-out test)

| class | detector | method | studies | rows | Spearman [95% CI] | Fisher-z rho | Pearson | RMSE log10 | AUROC<=500nM | bal.acc | Spearman gap vs Presto [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I | fluorescence | presto | 2 | 450 | 0.398 [0.211, 0.585] | 0.541 | 0.435 | 1.107 | 0.800 | 0.573 |  |
| I | fluorescence | netmhcpan41 | 2 | 450 | 0.700 [0.677, 0.724] | 0.684 | 0.695 | 0.885 | 0.789 | 0.592 | +0.302 [+0.092, +0.513] |
| I | fluorescence | mhcflurry | 2 | 450 | 0.582 [0.513, 0.652] | 0.534 | 0.572 | 1.006 | 0.759 | 0.549 | +0.184 [-0.073, +0.441] |
| I | fluorescence | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.428 | 0.500 | 0.500 |  |
| I | radioactivity | presto | 2 | 692 | 0.501 [0.302, 0.700] | 0.657 | 0.505 | 1.066 | 0.816 | 0.681 |  |
| I | radioactivity | netmhcpan41 | 2 | 692 | 0.638 [0.455, 0.822] | 0.788 | 0.625 | 0.922 | 0.859 | 0.772 | +0.137 [+0.122, +0.152] |
| I | radioactivity | mhcflurry | 2 | 692 | 0.580 [0.369, 0.791] | 0.752 | 0.546 | 1.151 | 0.782 | 0.640 | +0.079 [+0.067, +0.091] |
| I | radioactivity | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.550 | 0.500 | 0.500 |  |
| II | fluorescence | presto | 3 | 314 | 0.553 [0.504, 0.596] | 0.537 | 0.566 | 0.737 | 0.770 | 0.657 |  |
| II | fluorescence | netmhciipan43 | 3 | 314 | 0.766 [0.686, 0.867] | 0.810 | 0.784 | 0.724 | 0.910 | 0.724 | +0.213 [+0.090, +0.308] |
| II | fluorescence | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.131 | 0.500 | 0.500 |  |
| II | radioactivity | presto | 1 | 178 | 0.283 [0.283, 0.283] | 0.300 | 0.289 | 1.066 | 0.704 | 0.460 |  |
| II | radioactivity | netmhciipan43 | 1 | 178 | 0.434 [0.434, 0.434] | 0.557 | 0.422 | 0.614 | 0.861 | 0.570 | +0.151 [+0.151, +0.151] |
| II | radioactivity | train_mean | 0 | 0 | undefined (constant) | nan | nan | 0.949 | 0.500 | 0.250 |  |

### curated-multi-s43_validation (DEVELOPMENTAL (validation))

| class | detector | method | studies | rows | Spearman [95% CI] | Fisher-z rho | Pearson | RMSE log10 | AUROC<=500nM | bal.acc | Spearman gap vs Presto [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I | fluorescence | presto | 2 | 257 | 0.588 [0.583, 0.592] | 0.587 | 0.577 | 1.100 | 0.791 | 0.593 |  |
| I | fluorescence | netmhcpan41 | 2 | 257 | 0.724 [0.619, 0.829] | 0.760 | 0.712 | 0.800 | 0.955 | 0.658 | +0.137 [+0.027, +0.246] |
| I | fluorescence | mhcflurry | 2 | 257 | 0.635 [0.574, 0.696] | 0.650 | 0.614 | 0.896 | 0.884 | 0.625 | +0.047 [-0.018, +0.113] |
| I | fluorescence | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.203 | 0.500 | 0.500 |  |
| I | radioactivity | presto | 1 | 208 | 0.505 [0.505, 0.505] | 0.522 | 0.508 | 1.062 | nan | 0.391 |  |
| I | radioactivity | netmhcpan41 | 1 | 208 | 0.628 [0.628, 0.628] | 0.680 | 0.598 | 0.810 | nan | 0.432 | +0.123 [+0.123, +0.123] |
| I | radioactivity | mhcflurry | 1 | 208 | 0.625 [0.625, 0.625] | 0.667 | 0.613 | 0.923 | nan | 0.402 | +0.120 [+0.120, +0.120] |
| I | radioactivity | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.616 | nan | 0.000 |  |
| II | fluorescence | presto | 1 | 352 | 0.601 [0.601, 0.601] | 0.601 | 0.615 | 0.583 | 0.834 | 0.653 |  |
| II | fluorescence | netmhciipan43 | 1 | 352 | 0.859 [0.859, 0.859] | 0.859 | 0.854 | 0.401 | 0.943 | 0.803 | +0.258 [+0.258, +0.258] |
| II | fluorescence | train_mean | 0 | 0 | undefined (constant) | nan | nan | 0.782 | 0.500 | 0.500 |  |
| II | radioactivity | presto | 1 | 1375 | 0.315 [0.315, 0.315] | 0.437 | 0.331 | 0.995 | 0.752 | 0.542 |  |
| II | radioactivity | netmhciipan43 | 1 | 1375 | 0.460 [0.460, 0.460] | 0.701 | 0.468 | 0.635 | 0.900 | 0.665 | +0.145 [+0.145, +0.145] |
| II | radioactivity | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.240 | 0.500 | 0.250 |  |

### curated-multi-s42_test (held-out test)

| class | detector | method | studies | rows | Spearman [95% CI] | Fisher-z rho | Pearson | RMSE log10 | AUROC<=500nM | bal.acc | Spearman gap vs Presto [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I | fluorescence | presto | 2 | 450 | 0.408 [0.246, 0.569] | 0.530 | 0.417 | 1.108 | 0.670 | 0.561 |  |
| I | fluorescence | netmhcpan41 | 2 | 450 | 0.700 [0.677, 0.724] | 0.684 | 0.695 | 0.885 | 0.789 | 0.592 | +0.293 [+0.108, +0.478] |
| I | fluorescence | mhcflurry | 2 | 450 | 0.582 [0.513, 0.652] | 0.534 | 0.572 | 1.006 | 0.759 | 0.549 | +0.175 [-0.057, +0.406] |
| I | fluorescence | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.428 | 0.500 | 0.500 |  |
| I | radioactivity | presto | 2 | 692 | 0.504 [0.376, 0.633] | 0.602 | 0.520 | 1.224 | 0.853 | 0.780 |  |
| I | radioactivity | netmhcpan41 | 2 | 692 | 0.638 [0.455, 0.822] | 0.788 | 0.625 | 0.922 | 0.859 | 0.772 | +0.134 [+0.079, +0.189] |
| I | radioactivity | mhcflurry | 2 | 692 | 0.580 [0.369, 0.791] | 0.752 | 0.546 | 1.151 | 0.782 | 0.640 | +0.076 [-0.007, +0.158] |
| I | radioactivity | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.550 | 0.500 | 0.500 |  |
| II | fluorescence | presto | 3 | 314 | 0.535 [0.468, 0.605] | 0.507 | 0.553 | 0.782 | 0.765 | 0.678 |  |
| II | fluorescence | netmhciipan43 | 3 | 314 | 0.766 [0.686, 0.867] | 0.810 | 0.784 | 0.724 | 0.910 | 0.724 | +0.231 [+0.081, +0.400] |
| II | fluorescence | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.131 | 0.500 | 0.500 |  |
| II | radioactivity | presto | 1 | 178 | 0.238 [0.238, 0.238] | 0.230 | 0.272 | 1.075 | 0.611 | 0.415 |  |
| II | radioactivity | netmhciipan43 | 1 | 178 | 0.434 [0.434, 0.434] | 0.557 | 0.422 | 0.614 | 0.861 | 0.570 | +0.197 [+0.197, +0.197] |
| II | radioactivity | train_mean | 0 | 0 | undefined (constant) | nan | nan | 0.949 | 0.500 | 0.250 |  |

### curated-multi-s42_validation (DEVELOPMENTAL (validation))

| class | detector | method | studies | rows | Spearman [95% CI] | Fisher-z rho | Pearson | RMSE log10 | AUROC<=500nM | bal.acc | Spearman gap vs Presto [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I | fluorescence | presto | 2 | 257 | 0.649 [0.641, 0.657] | 0.651 | 0.642 | 0.982 | 0.838 | 0.584 |  |
| I | fluorescence | netmhcpan41 | 2 | 257 | 0.724 [0.619, 0.829] | 0.760 | 0.712 | 0.800 | 0.955 | 0.658 | +0.075 [-0.022, +0.172] |
| I | fluorescence | mhcflurry | 2 | 257 | 0.635 [0.574, 0.696] | 0.650 | 0.614 | 0.896 | 0.884 | 0.625 | -0.014 [-0.067, +0.038] |
| I | fluorescence | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.203 | 0.500 | 0.500 |  |
| I | radioactivity | presto | 1 | 208 | 0.480 [0.480, 0.480] | 0.518 | 0.479 | 1.087 | nan | 0.373 |  |
| I | radioactivity | netmhcpan41 | 1 | 208 | 0.628 [0.628, 0.628] | 0.680 | 0.598 | 0.810 | nan | 0.432 | +0.148 [+0.148, +0.148] |
| I | radioactivity | mhcflurry | 1 | 208 | 0.625 [0.625, 0.625] | 0.667 | 0.613 | 0.923 | nan | 0.402 | +0.145 [+0.145, +0.145] |
| I | radioactivity | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.616 | nan | 0.000 |  |
| II | fluorescence | presto | 1 | 352 | 0.606 [0.606, 0.606] | 0.606 | 0.617 | 0.627 | 0.824 | 0.694 |  |
| II | fluorescence | netmhciipan43 | 1 | 352 | 0.859 [0.859, 0.859] | 0.859 | 0.854 | 0.401 | 0.943 | 0.803 | +0.253 [+0.253, +0.253] |
| II | fluorescence | train_mean | 0 | 0 | undefined (constant) | nan | nan | 0.782 | 0.500 | 0.500 |  |
| II | radioactivity | presto | 1 | 1375 | 0.310 [0.310, 0.310] | 0.464 | 0.326 | 1.041 | 0.769 | 0.549 |  |
| II | radioactivity | netmhciipan43 | 1 | 1375 | 0.460 [0.460, 0.460] | 0.701 | 0.468 | 0.635 | 0.900 | 0.665 | +0.149 [+0.149, +0.149] |
| II | radioactivity | train_mean | 0 | 0 | undefined (constant) | nan | nan | 1.240 | 0.500 | 0.250 |  |
