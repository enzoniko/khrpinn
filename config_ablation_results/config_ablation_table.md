# Configuration Ablation Study

Results with 95% Bootstrap Confidence Intervals (N=100 seeds)

## Duffing System

| Configuration | Test MSE | Discovery R² |
|---------------|----------|--------------|
| Config A (Baseline) | 0.000541 ± 0.000067 | 0.835088 ± 0.029615 |
| Config B (Spline-Forced) | 0.000542 ± 0.000067 | 0.773073 ± 0.096997 |
| Config C (Sparse-Low) | 0.000543 ± 0.000067 | 0.594780 ± 0.032735 |
| Config D (Sparse-High) | 0.000543 ± 0.000067 | 0.581518 ± 0.026408 |
| Config E (Aggressive-Grid) | 0.000541 ± 0.000067 | 0.794101 ± 0.066953 |
| Config F (Coarse-Grid) | 0.000542 ± 0.000067 | 0.862024 ± 0.036667 |
| Config G (Fine-Grid) | 0.000548 ± 0.000066 | 0.745094 ± 0.099298 |

## VdP System

| Configuration | Test MSE | Discovery R² |
|---------------|----------|--------------|
| Config A (Baseline) | 0.000908 ± 0.000148 | 0.667087 ± 0.037017 |
| Config B (Spline-Forced) | 0.000906 ± 0.000149 | 0.319809 ± 0.329899 |
| Config C (Sparse-Low) | 0.000959 ± 0.000146 | -5.229273 ± 5.090545 |
| Config D (Sparse-High) | 0.000972 ± 0.000149 | -1.687973 ± 1.317942 |
| Config E (Aggressive-Grid) | 0.000908 ± 0.000148 | 0.698976 ± 0.064810 |
| Config F (Coarse-Grid) | 0.000908 ± 0.000147 | 0.638524 ± 0.302357 |
| Config G (Fine-Grid) | 0.000915 ± 0.000154 | -0.174154 ± 0.691320 |

