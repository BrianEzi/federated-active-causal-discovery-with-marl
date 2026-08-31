# Sweep results — results/sweep/oracle

20 of 20 cells have at least one seed; 60 runs total.

CONFIG (constant across cells): evidence=oracle, turn_aware_credit=True, local_epochs=4, n_int=20, evaluation=sampled policy over 200 episodes.
GATE: mean window rate over the last 10 checkpoints must be >= 0.7. A run below it never learned to solve windows and its score says nothing about the task.
The MI ratio is still shown per cell but NO LONGER GATES: it tracks final entropy, and a policy can solve 100% of windows without conditioning on state where the budget is generous. The old MI floor of 0.15 discarded two entire cells in which every seed solved 95-100% of windows.

### k axis — hard SHD of the pooled global graph (LOWER IS BETTER)
cell                        k seeds           learned   greedy   random     L/G  resolved  excluded
k04s50n04b150               4     3   0.0117+- 0.0034   0.0061   0.0886    1.91     0.959  
k08s50n04b150               8     3   0.0011+- 0.0010   0.0008   0.0541    1.30     0.970  
k12s50n04b150              12     3   0.0001+- 0.0001   0.0008   0.0391    0.10     0.972  
k20s50n04b150              20     2   0.0000+- 0.0000   0.0006   0.0247    0.08     0.973  [2]
k30s50n04b150              30     1   0.0001+- 0.0000   0.0005   0.0177    0.10     0.975  [1, 2]

### k axis — episode success (conjunction)
cell                        k seeds         learned  greedy             gap  ceiling     MI       steps L/G/C  gate
k04s50n04b150               4     3   0.808+-0.044   0.883  -0.075+-0.075    1.000  0.514  3.28/ 2.98/ 2.52  ok
k08s50n04b150               8     3   0.922+-0.060   0.947  -0.025+-0.056    1.000  0.365  5.26/ 5.02/ 4.40  ok
k12s50n04b150              12     3   0.977+-0.013   0.918  +0.058+-0.013    1.000  0.342  6.64/ 7.13/ 6.22  ok
k20s50n04b150              20     3   0.780+-0.346   0.897  -0.117+-0.358    1.000  0.248 12.76/10.60/ 9.88  window<0.7 seeds [2]  [low MI, kept: [2]]
k30s50n04b150              30     3   0.320+-0.537   0.843  -0.523+-0.579    1.000  0.100 21.70/15.34/14.46  window<0.7 seeds [1, 2]  [low MI, kept: [1, 2]]

### sigma axis — hard SHD of the pooled global graph (LOWER IS BETTER)
cell                    sigma seeds           learned   greedy   random     L/G  resolved  excluded
k12s25n04b150            0.25     2   0.0002+- 0.0002   0.0010   0.0634    0.22     0.994  [2]
k12s50n04b150             0.5     3   0.0001+- 0.0001   0.0008   0.0391    0.10     0.972  
k12s75n04b150            0.75     3   0.0017+- 0.0025   0.0004   0.0284    4.36     0.923  

### sigma axis — episode success (conjunction)
cell                    sigma seeds         learned  greedy             gap  ceiling     MI       steps L/G/C  gate
k12s25n04b150            0.25     3   0.805+-0.279   0.892  -0.087+-0.286    1.000  0.364  8.56/ 8.32/ 5.62  window<0.7 seeds [2]
k12s50n04b150             0.5     3   0.977+-0.013   0.918  +0.058+-0.013    1.000  0.342  6.64/ 7.13/ 6.22  ok
k12s75n04b150            0.75     3   0.870+-0.182   0.972  -0.102+-0.181    1.000  0.236  7.00/ 5.57/ 6.93  ok  [low MI, kept: [2]]

### n axis — hard SHD of the pooled global graph (LOWER IS BETTER)
cell                        n seeds           learned   greedy   random     L/G  resolved  excluded
k12s50n02b150               2     2   0.0002+- 0.0002   0.0015   0.0671    0.12     0.938  [2]
k12s50n03b150               3     2   0.0002+- 0.0002   0.0007   0.0474    0.33     0.960  [2]
k12s50n04b150               4     3   0.0001+- 0.0001   0.0008   0.0391    0.10     0.972  
k12s50n05b150               5     3   0.0005+- 0.0007   0.0003   0.0325    1.65     0.978  
k12s50n08b150               8     3   0.0012+- 0.0015   0.0003   0.0252    4.24     0.987  
k12s50n10b150              10     3   0.0009+- 0.0012   0.0001   0.0211    6.75     0.991  

### n axis — episode success (conjunction)
cell                        n seeds         learned  greedy             gap  ceiling     MI       steps L/G/C  gate
k12s50n02b150               2     3   0.740+-0.433   0.907  -0.167+-0.427    1.000  0.402  9.04/ 8.20/ 7.09  window<0.7 seeds [2]
k12s50n03b150               3     3   0.818+-0.276   0.945  -0.127+-0.281    1.000  0.344  8.07/ 7.20/ 6.58  window<0.7 seeds [2]  [low MI, kept: [2]]
k12s50n04b150               4     3   0.977+-0.013   0.918  +0.058+-0.013    1.000  0.342  6.64/ 7.13/ 6.22  ok
k12s50n05b150               5     3   0.902+-0.140   0.957  -0.055+-0.139    1.000  0.284  7.32/ 6.79/ 5.96  ok  [low MI, kept: [2]]
k12s50n08b150               8     3   0.812+-0.154   0.940  -0.128+-0.128    1.000  0.235  8.03/ 6.83/ 5.37  ok  [low MI, kept: [2]]
k12s50n10b150              10     3   0.818+-0.181   0.968  -0.150+-0.160    1.000  0.200  8.21/ 6.56/ 5.10  ok  [low MI, kept: [2]]

### beta axis — hard SHD of the pooled global graph (LOWER IS BETTER)
cell                     beta seeds           learned   greedy   random     L/G  resolved  excluded
k12s50n04b100             1.0     2   0.0006+- 0.0002   0.0013   0.0585    0.42     0.971  [2]
k12s50n04b120             1.2     2   0.0003+- 0.0002   0.0009   0.0508    0.30     0.972  [2]
k12s50n04b150             1.5     3   0.0001+- 0.0001   0.0008   0.0391    0.10     0.972  
k12s50n04b200             2.0     3   0.0005+- 0.0006   0.0006   0.0246    0.84     0.971  
k12s50n04b500             5.0     3   0.0002+- 0.0002   0.0006   0.0024    0.40     0.972  

### beta axis — episode success (conjunction)
cell                     beta seeds         learned  greedy             gap  ceiling     MI       steps L/G/C  gate
k12s50n04b100             1.0     3   0.670+-0.450   0.815  -0.145+-0.438    0.997  0.304  6.71/ 6.56/ 6.22  window<0.7 seeds [2]  [low MI, kept: [2]]
k12s50n04b120             1.2     3   0.817+-0.240   0.878  -0.062+-0.234    1.000  0.333  6.79/ 6.78/ 6.22  window<0.7 seeds [2]  [low MI, kept: [2]]
k12s50n04b150             1.5     3   0.977+-0.013   0.918  +0.058+-0.013    1.000  0.342  6.64/ 7.13/ 6.22  ok
k12s50n04b200             2.0     3   0.917+-0.110   0.947  -0.030+-0.104    1.000  0.268  8.54/ 7.33/ 6.22  ok  [low MI, kept: [2]]
k12s50n04b500             5.0     3   0.948+-0.045   0.947  +0.002+-0.039    1.000  0.045 21.06/ 8.62/ 6.22  ok  [low MI, kept: [0, 1, 2]]

### n_int axis — hard SHD of the pooled global graph (LOWER IS BETTER)
cell                    n_int seeds           learned   greedy   random     L/G  resolved  excluded
k12s50n04b150              20     3   0.0001+- 0.0001   0.0008   0.0391    0.10     0.972  

### n_int axis — episode success (conjunction)
cell                    n_int seeds         learned  greedy             gap  ceiling     MI       steps L/G/C  gate
k12s50n04b150              20     3   0.977+-0.013   0.918  +0.058+-0.013    1.000  0.342  6.64/ 7.13/ 6.22  ok

### sigma_x_n axis — hard SHD of the pooled global graph (LOWER IS BETTER)
cell                    sigma seeds           learned   greedy   random     L/G  resolved  excluded
k12s25n02b150            0.25     2   0.0002+- 0.0001   0.0024   0.0818    0.06     0.987  [2]
k12s25n08b150            0.25     2   0.0007+- 0.0001   0.0007   0.0495    1.09     0.996  [2]
k12s75n02b150            0.75     3   0.0013+- 0.0021   0.0007   0.0580    1.87     0.867  
k12s75n08b150            0.75     3   0.0008+- 0.0009   0.0003   0.0207    3.02     0.962  

### sigma_x_n axis — episode success (conjunction)
cell                    sigma seeds         learned  greedy             gap  ceiling     MI       steps L/G/C  gate
k12s25n02b150            0.25     3   0.667+-0.465   0.838  -0.172+-0.454    1.000  0.467  9.15/ 8.77/ 6.51  window<0.7 seeds [2]
k12s25n08b150            0.25     3   0.585+-0.477   0.833  -0.248+-0.456    1.000  0.197 10.31/ 8.98/ 5.16  window<0.7 seeds [2]  [low MI, kept: [2]]
k12s75n02b150            0.75     3   0.842+-0.196   0.958  -0.117+-0.206    1.000  0.252  8.87/ 7.50/ 7.45  ok  [low MI, kept: [2]]
k12s75n08b150            0.75     3   0.885+-0.091   0.965  -0.080+-0.074    1.000  0.093  7.80/ 5.26/ 6.53  ok  [low MI, kept: [0, 1, 2]]

### Headline, over the 50 runs that clear the competence gate
  learned - greedy   -0.016 +- 0.118   (beats greedy in 28/50 runs)
  ceiling - learned  +0.095 +- 0.104   (headroom left to the optimal arm)
  EXCLUDED: 10 run(s) below the competence gate, listed per-axis above.
  wrote results/figures/axis_k.png
  wrote results/figures/axis_n.png
  wrote results/figures/axis_beta.png
  wrote results/figures/axis_sigma.png
