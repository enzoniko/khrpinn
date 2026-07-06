# Parameter Efficiency Ablation Study with BPTT - Summary Table

Results with Mean ± 95% CI (N=5 seeds)

## Duffing System

| Architecture | Configuration | Parameters | Discovery R² |
|--------------|----------------|------------|--------------|
| KAN | KAN Deep | 880 | -1607722.506795 ± 3151126.664142 |
| KAN | KAN Small | 240 | 0.873882 ± 0.079678 |
| KAN | KAN Very Small | 120 | 0.914165 ± 0.061435 |
| KAN | KAN Wide | 480 | 0.467607 ± 0.772653 |
| MLP | MLP Large | 4417 | 0.932407 ± 0.063263 |
| MLP | MLP Medium | 1185 | 0.951139 ± 0.032565 |
| MLP | MLP Small | 337 | 0.937100 ± 0.046905 |
| MLP | MLP Tiny | 105 | 0.905780 ± 0.092183 |

## VdP System

| Architecture | Configuration | Parameters | Discovery R² |
|--------------|----------------|------------|--------------|
| KAN | KAN Deep | 880 | 0.754081 ± 0.078674 |
| KAN | KAN Small | 240 | 0.785211 ± 0.072776 |
| KAN | KAN Very Small | 120 | 0.743317 ± 0.061094 |
| KAN | KAN Wide | 480 | -0.601847 ± 2.841727 |
| MLP | MLP Large | 4417 | 0.897635 ± 0.016742 |
| MLP | MLP Medium | 1185 | 0.878624 ± 0.019416 |
| MLP | MLP Small | 337 | 0.879107 ± 0.031796 |
| MLP | MLP Tiny | 105 | 0.621986 ± 0.173196 |

