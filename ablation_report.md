# Ablation Study: Linear Bias & Grid Starvation

Results with 95% Bootstrap Confidence Intervals (N=5 seeds)

## Duffing System

| Configuration | Test MSE | Discovery R² |
|---------------|----------|--------------|
| Config A (Baseline) | 0.000704 ± 0.000346 | 0.823711 ± 0.145731 |
| Config B (Spline-Forced) | 0.000703 ± 0.000346 | 0.864074 ± 0.084329 |
| Config C (Sparse-Symbolic) | 0.000706 ± 0.000348 | 0.568302 ± 0.141677 |

## VdP System

| Configuration | Test MSE | Discovery R² |
|---------------|----------|--------------|
| Config A (Baseline) | 0.000757 ± 0.000608 | 0.675348 ± 0.094772 |
| Config B (Spline-Forced) | 0.000753 ± 0.000612 | 0.682441 ± 0.146155 |
| Config C (Sparse-Symbolic) | 0.000799 ± 0.000600 | -0.007481 ± 0.410585 |

