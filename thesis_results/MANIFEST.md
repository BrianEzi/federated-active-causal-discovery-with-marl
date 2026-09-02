# Thesis results — the files the dissertation quotes

Built by `scripts/collect_thesis_results.py`. **Do not edit by hand.** To add a
result, add it to that script's `REGISTRY` so the claim it supports and the
command that regenerates it travel with the data.

`python scripts/collect_thesis_results.py --check` verifies that no source has
been re-run since these copies were taken.

## `sweep/` — 4.2 (RQ1), figure sweep_grid

Learned against myopic and random on all four swept axes: window size, agent count, contended fraction, budget. 20 cells x 3 seeds. Final-policy evaluation.

Regenerate with: `scripts/ma_train.py per results/sweep/oracle/jobs/*.sh`

| file | sha256 (16) | source |
|---|---|---|
| `k04s50n04b150_s0.json` | `440ce5ad37ba8c64` | `results/sweep/oracle/k04s50n04b150_s0.json` |
| `k04s50n04b150_s1.json` | `7d47c05ec8680b86` | `results/sweep/oracle/k04s50n04b150_s1.json` |
| `k04s50n04b150_s2.json` | `accb425a73211158` | `results/sweep/oracle/k04s50n04b150_s2.json` |
| `k08s50n04b150_s0.json` | `90e7c9ff2d390f75` | `results/sweep/oracle/k08s50n04b150_s0.json` |
| `k08s50n04b150_s1.json` | `d05a6210979af070` | `results/sweep/oracle/k08s50n04b150_s1.json` |
| `k08s50n04b150_s2.json` | `bebbf455c0c399f7` | `results/sweep/oracle/k08s50n04b150_s2.json` |
| `k12s25n02b150_s0.json` | `af8cf20866800692` | `results/sweep/oracle/k12s25n02b150_s0.json` |
| `k12s25n02b150_s1.json` | `dac7277e2ad7d7b7` | `results/sweep/oracle/k12s25n02b150_s1.json` |
| `k12s25n02b150_s2.json` | `7c163c43fcb07448` | `results/sweep/oracle/k12s25n02b150_s2.json` |
| `k12s25n04b150_s0.json` | `d38490fe7ac189a5` | `results/sweep/oracle/k12s25n04b150_s0.json` |
| `k12s25n04b150_s1.json` | `832247592ed182c9` | `results/sweep/oracle/k12s25n04b150_s1.json` |
| `k12s25n04b150_s2.json` | `d2e866bb133b1ccc` | `results/sweep/oracle/k12s25n04b150_s2.json` |
| `k12s25n08b150_s0.json` | `fb3e89acf9496ec9` | `results/sweep/oracle/k12s25n08b150_s0.json` |
| `k12s25n08b150_s1.json` | `960432c7f5e63745` | `results/sweep/oracle/k12s25n08b150_s1.json` |
| `k12s25n08b150_s2.json` | `71f7413a95854dbd` | `results/sweep/oracle/k12s25n08b150_s2.json` |
| `k12s50n02b150_s0.json` | `4f675e41643cff65` | `results/sweep/oracle/k12s50n02b150_s0.json` |
| `k12s50n02b150_s1.json` | `c39b407b88200dd4` | `results/sweep/oracle/k12s50n02b150_s1.json` |
| `k12s50n02b150_s2.json` | `72556be2dc05b4c7` | `results/sweep/oracle/k12s50n02b150_s2.json` |
| `k12s50n03b150_s0.json` | `30bb2365cf72d4b1` | `results/sweep/oracle/k12s50n03b150_s0.json` |
| `k12s50n03b150_s1.json` | `572b6d167f2aa74c` | `results/sweep/oracle/k12s50n03b150_s1.json` |
| `k12s50n03b150_s2.json` | `6a4a43f6113e648c` | `results/sweep/oracle/k12s50n03b150_s2.json` |
| `k12s50n04b100_s0.json` | `cd4621a71556d99b` | `results/sweep/oracle/k12s50n04b100_s0.json` |
| `k12s50n04b100_s1.json` | `01e82a51eb6446ae` | `results/sweep/oracle/k12s50n04b100_s1.json` |
| `k12s50n04b100_s2.json` | `dd9c72e0d98c968f` | `results/sweep/oracle/k12s50n04b100_s2.json` |
| `k12s50n04b120_s0.json` | `fbfb926ed1de4fe9` | `results/sweep/oracle/k12s50n04b120_s0.json` |
| `k12s50n04b120_s1.json` | `0dc83a3292f52a25` | `results/sweep/oracle/k12s50n04b120_s1.json` |
| `k12s50n04b120_s2.json` | `098773d28a3d2ad5` | `results/sweep/oracle/k12s50n04b120_s2.json` |
| `k12s50n04b150_s0.json` | `d350f4937d881fbd` | `results/sweep/oracle/k12s50n04b150_s0.json` |
| `k12s50n04b150_s1.json` | `c1ee4c231c098bfc` | `results/sweep/oracle/k12s50n04b150_s1.json` |
| `k12s50n04b150_s2.json` | `af90b30eaf51ad2f` | `results/sweep/oracle/k12s50n04b150_s2.json` |
| `k12s50n04b200_s0.json` | `1ce0cea1189153e0` | `results/sweep/oracle/k12s50n04b200_s0.json` |
| `k12s50n04b200_s1.json` | `b1cd0d0fb218d014` | `results/sweep/oracle/k12s50n04b200_s1.json` |
| `k12s50n04b200_s2.json` | `12ef397f615c265e` | `results/sweep/oracle/k12s50n04b200_s2.json` |
| `k12s50n04b500_s0.json` | `65deb368daf83810` | `results/sweep/oracle/k12s50n04b500_s0.json` |
| `k12s50n04b500_s1.json` | `91b108e406dea139` | `results/sweep/oracle/k12s50n04b500_s1.json` |
| `k12s50n04b500_s2.json` | `517d926d40e21a5b` | `results/sweep/oracle/k12s50n04b500_s2.json` |
| `k12s50n05b150_s0.json` | `6142c18a7c4cc934` | `results/sweep/oracle/k12s50n05b150_s0.json` |
| `k12s50n05b150_s1.json` | `9204725300a48d92` | `results/sweep/oracle/k12s50n05b150_s1.json` |
| `k12s50n05b150_s2.json` | `33310e5049312028` | `results/sweep/oracle/k12s50n05b150_s2.json` |
| `k12s50n08b150_s0.json` | `6a6b9b3633cd6035` | `results/sweep/oracle/k12s50n08b150_s0.json` |
| `k12s50n08b150_s1.json` | `3cfa67442800bb61` | `results/sweep/oracle/k12s50n08b150_s1.json` |
| `k12s50n08b150_s2.json` | `80406269b9992803` | `results/sweep/oracle/k12s50n08b150_s2.json` |
| `k12s50n10b150_s0.json` | `aa869e0b7269af80` | `results/sweep/oracle/k12s50n10b150_s0.json` |
| `k12s50n10b150_s1.json` | `978de48b5c2e4092` | `results/sweep/oracle/k12s50n10b150_s1.json` |
| `k12s50n10b150_s2.json` | `018acf72f72b457f` | `results/sweep/oracle/k12s50n10b150_s2.json` |
| `k12s75n02b150_s0.json` | `dfea209e826f320b` | `results/sweep/oracle/k12s75n02b150_s0.json` |
| `k12s75n02b150_s1.json` | `8fc255f5c3873833` | `results/sweep/oracle/k12s75n02b150_s1.json` |
| `k12s75n02b150_s2.json` | `eaa42c5500aa7755` | `results/sweep/oracle/k12s75n02b150_s2.json` |
| `k12s75n04b150_s0.json` | `d99342ae02b72028` | `results/sweep/oracle/k12s75n04b150_s0.json` |
| `k12s75n04b150_s1.json` | `eac4524e2e435f23` | `results/sweep/oracle/k12s75n04b150_s1.json` |
| `k12s75n04b150_s2.json` | `f2b8b54b98731c8a` | `results/sweep/oracle/k12s75n04b150_s2.json` |
| `k12s75n08b150_s0.json` | `198cc3584f954eca` | `results/sweep/oracle/k12s75n08b150_s0.json` |
| `k12s75n08b150_s1.json` | `6d7aaa0ec49632a3` | `results/sweep/oracle/k12s75n08b150_s1.json` |
| `k12s75n08b150_s2.json` | `33cf60463bac56c2` | `results/sweep/oracle/k12s75n08b150_s2.json` |
| `k20s50n04b150_s0.json` | `6c509d07702c042a` | `results/sweep/oracle/k20s50n04b150_s0.json` |
| `k20s50n04b150_s1.json` | `ad16675c3d444847` | `results/sweep/oracle/k20s50n04b150_s1.json` |
| `k20s50n04b150_s2.json` | `07d7ee6f31b64905` | `results/sweep/oracle/k20s50n04b150_s2.json` |
| `k30s50n04b150_s0.json` | `b997f4f8325f5653` | `results/sweep/oracle/k30s50n04b150_s0.json` |
| `k30s50n04b150_s1.json` | `1a3fe20cc7dcbb58` | `results/sweep/oracle/k30s50n04b150_s1.json` |
| `k30s50n04b150_s2.json` | `7bd9861426b6ed96` | `results/sweep/oracle/k30s50n04b150_s2.json` |

60 files.

## `checkpoint/` — 4.1.1 and 4.2 (RQ1), figure checkpoint

Early-stopped against final policy on the window-size axis, 200 paired episodes per seed. Establishes that the checkpoint choice is inert below the crossover and worth 2.3x at k=20 and 16x at k=30 above it.

Regenerate with: `scripts/global_shd_paired.py --episodes 200 --sample --checkpoint {best,final}`

| file | sha256 (16) | source |
|---|---|---|
| `k04_best.json` | `7e59d41377929726` | `results/rerows/k04_best.json` |
| `k08_best.json` | `2df3e903d5bff2d1` | `results/rerows/k08_best.json` |
| `k12_best.json` | `42df423b453e4b19` | `results/rerows/k12_best.json` |
| `k20_best.json` | `3743b81d763ae208` | `results/rerows/k20_best.json` |
| `k30_best.json` | `257e8832c8ed424d` | `results/rerows/k30_best.json` |
| `k04_final.json` | `e072b5ce034cfe74` | `results/rerows/k04_final.json` |
| `k08_final.json` | `5efeb63aa2ae1e86` | `results/rerows/k08_final.json` |
| `k12_final.json` | `207afe020cc42188` | `results/rerows/k12_final.json` |
| `k20_final.json` | `eee0667433e2bdb9` | `results/rerows/k20_final.json` |
| `k30_final.json` | `53b5d0ffc704c815` | `results/rerows/k30_final.json` |

10 files.

## `attribution/` — RQ4, figure attribution_law

RQ4. attr_ceiling: recovery by group size and peer count. attr_ceiling_matched_budget: the control holding rounds-per-agent fixed, which is what rules out budget starvation. attr_ceiling_budget: the coverage step function (21/1056 at budget 30; 349/1056 at 60, 120 and 240 -- IDENTICAL counts, not merely equal rates). attr_scale_final and attr_reach: k=30/40/50 at 30 episodes each, zero misattributions. attr_train: training under the ATTRIBUTION BELIEF BACKEND, scored on the structural criterion -- NOT an attribution reward, which no run in this project uses. attr/transfer_*: the self-interested attribution baseline.

Regenerate with: `scripts/attr_ceiling.py, scripts/attr_model.py`

| file | sha256 (16) | source |
|---|---|---|
| `attr_ceiling.json` | `9c569a531f2755d1` | `results/attr_ceiling.json` |
| `attr_ceiling_matched_budget.json` | `9b7ecc85010f907e` | `results/attr_ceiling_matched_budget.json` |
| `attr_ceiling_budget.json` | `f8126f68dfbbcf2a` | `results/attr_ceiling_budget.json` |
| `attr_scale_final.json` | `ab9fa47f295d6ffd` | `results/attr_scale_final.json` |
| `attr_reach.json` | `0aaa67772678da9b` | `results/attr_reach.json` |
| `k12s50n04b200_attr_s0.json` | `b9b32316357e6d31` | `results/attr_train/k12s50n04b200_attr_s0.json` |
| `k12s50n04b200_attr_s1.json` | `5518f1f2bed61742` | `results/attr_train/k12s50n04b200_attr_s1.json` |
| `k12s50n04b200_attr_s2.json` | `1d268dbf91c21499` | `results/attr_train/k12s50n04b200_attr_s2.json` |
| `transfer_k12s50n04b200_s0.json` | `10decb72913ca1a3` | `results/attr/transfer_k12s50n04b200_s0.json` |
| `transfer_k12s50n04b200_s1.json` | `8f9a655be18a4e90` | `results/attr/transfer_k12s50n04b200_s1.json` |
| `transfer_k12s50n04b200_s2.json` | `4242e871744fd33b` | `results/attr/transfer_k12s50n04b200_s2.json` |

11 files.

## `federation/` — 4.4 (RQ3), figure federation

RQ3, plus the sweep's training-budget limitation. v2_k12_* and v2_k20_*: arm A is the federated baseline, arm E removes the information partition (partners' beliefs and counts observed) and the optimiser partition (trajectories pooled instead of FedAvg). Action rights stay partitioned in both. shd_k20_*: the same arms on the primary metric, where the recovery rate has saturated and cannot separate them. longcheck/*_long_s2: all seven competence-floor exclusions retrained at 12,000 episodes; all seven pass and all seven beat the myopic rule. lrcheck/*: the same runs at lr 1e-4, which makes them worse and rules out an unstable step size.

Regenerate with: `results/central/jobs/*.sh and jobs2/*.sh, then global_shd_paired.py`

| file | sha256 (16) | source |
|---|---|---|
| `v2_k12_A_s0.json` | `224100d5e00d9bbb` | `results/central/v2_k12_A_s0.json` |
| `v2_k12_A_s1.json` | `327c704b4029933d` | `results/central/v2_k12_A_s1.json` |
| `v2_k12_A_s2.json` | `9b58fbc3f1687107` | `results/central/v2_k12_A_s2.json` |
| `v2_k12_A_s3.json` | `a19b02b928f3d655` | `results/central/v2_k12_A_s3.json` |
| `v2_k12_A_s4.json` | `63b1370264960ede` | `results/central/v2_k12_A_s4.json` |
| `v2_k12_A_s5.json` | `fb5d1d4af592cda3` | `results/central/v2_k12_A_s5.json` |
| `v2_k12_B_s0.json` | `e73fd97c508bc8e9` | `results/central/v2_k12_B_s0.json` |
| `v2_k12_B_s1.json` | `57dd04ccf0e63146` | `results/central/v2_k12_B_s1.json` |
| `v2_k12_B_s2.json` | `e47a1d923f79657c` | `results/central/v2_k12_B_s2.json` |
| `v2_k12_E_s0.json` | `30ca8a38bae742a6` | `results/central/v2_k12_E_s0.json` |
| `v2_k12_E_s1.json` | `305074dc4c7a7c69` | `results/central/v2_k12_E_s1.json` |
| `v2_k12_E_s2.json` | `66c1846446e9bd5c` | `results/central/v2_k12_E_s2.json` |
| `v2_k12_E_s3.json` | `295e939d0531f30f` | `results/central/v2_k12_E_s3.json` |
| `v2_k12_E_s4.json` | `96136efb22b7a9b3` | `results/central/v2_k12_E_s4.json` |
| `v2_k12_E_s5.json` | `9845cb47fc21fed5` | `results/central/v2_k12_E_s5.json` |
| `v2_k20_A_s0.json` | `90a4228cdbe9a8ee` | `results/central/v2_k20_A_s0.json` |
| `v2_k20_A_s1.json` | `4cd4ccd0a0de7d5b` | `results/central/v2_k20_A_s1.json` |
| `v2_k20_A_s2.json` | `c0505184b28b21ae` | `results/central/v2_k20_A_s2.json` |
| `v2_k20_E_s0.json` | `da8fee69af83ca4a` | `results/central/v2_k20_E_s0.json` |
| `v2_k20_E_s1.json` | `49c26918edd99849` | `results/central/v2_k20_E_s1.json` |
| `v2_k20_E_s2.json` | `fa486406b54162fd` | `results/central/v2_k20_E_s2.json` |
| `shd_A.json` | `a97e605d7fc07b4d` | `results/rerows/shd_A.json` |
| `shd_E.json` | `bc7606b933ba2a2e` | `results/rerows/shd_E.json` |
| `shd_A_s345.json` | `1cbc0ba29c83d4f5` | `results/rerows/shd_A_s345.json` |
| `shd_E_s345.json` | `28ed3255a2c46d24` | `results/rerows/shd_E_s345.json` |
| `shd_k20_A.json` | `f4f99c518056ffb5` | `results/rerows/shd_k20_A.json` |
| `shd_k20_E.json` | `f9292a9dbf3d66a4` | `results/rerows/shd_k20_E.json` |
| `k12s25n02b150_long_s2.json` | `1cf9d31735174b67` | `results/longcheck/k12s25n02b150_long_s2.json` |
| `k12s25n04b150_long_s2.json` | `abc7c6b42e20421f` | `results/longcheck/k12s25n04b150_long_s2.json` |
| `k12s25n08b150_long_s2.json` | `3f80ee2d9aa2218d` | `results/longcheck/k12s25n08b150_long_s2.json` |
| `k12s25n08b150_long_s3.json` | `79c23e302cf40918` | `results/longcheck/k12s25n08b150_long_s3.json` |
| `k12s50n02b150_long_s2.json` | `5047e1d62ad567d2` | `results/longcheck/k12s50n02b150_long_s2.json` |
| `k12s50n03b150_long_s2.json` | `b28124e5fbdacaf6` | `results/longcheck/k12s50n03b150_long_s2.json` |
| `k12s50n04b100_long_s2.json` | `f0136cc95df44ef5` | `results/longcheck/k12s50n04b100_long_s2.json` |
| `k12s50n04b120_long_s2.json` | `d609f059d8e68849` | `results/longcheck/k12s50n04b120_long_s2.json` |
| `k12s50n05b150_conv_s0.json` | `226aac33595dfb3e` | `results/longcheck/k12s50n05b150_conv_s0.json` |
| `k12s50n05b150_conv_s1.json` | `9622e72de5ba93df` | `results/longcheck/k12s50n05b150_conv_s1.json` |
| `k12s50n05b150_conv_s2.json` | `b8f7f73d1bf5eb24` | `results/longcheck/k12s50n05b150_conv_s2.json` |
| `k12s50n08b150_conv_s0.json` | `0434b8f76e009ac6` | `results/longcheck/k12s50n08b150_conv_s0.json` |
| `k12s50n08b150_conv_s1.json` | `a67e01770683e95b` | `results/longcheck/k12s50n08b150_conv_s1.json` |
| `k12s50n08b150_conv_s2.json` | `159ee50fb9c70023` | `results/longcheck/k12s50n08b150_conv_s2.json` |
| `k12s50n10b150_conv_s0.json` | `2a5a2a22d5f80e27` | `results/longcheck/k12s50n10b150_conv_s0.json` |
| `k12s50n10b150_conv_s1.json` | `ac3f7b8895c4383c` | `results/longcheck/k12s50n10b150_conv_s1.json` |
| `k12s50n10b150_conv_s2.json` | `599f2b8b882c10af` | `results/longcheck/k12s50n10b150_conv_s2.json` |
| `k12s75n04b150_conv_s0.json` | `45a5f2528ee92438` | `results/longcheck/k12s75n04b150_conv_s0.json` |
| `k12s75n04b150_conv_s1.json` | `273bc5ac56b35457` | `results/longcheck/k12s75n04b150_conv_s1.json` |
| `k12s75n04b150_conv_s2.json` | `1041c0c4f3603aa3` | `results/longcheck/k12s75n04b150_conv_s2.json` |
| `k12s25n08b150_lr1e4_s2.json` | `8c4173de43df08e4` | `results/lrcheck/k12s25n08b150_lr1e4_s2.json` |
| `k12s25n08b150_lr1e4_s3.json` | `c8b028825f007ab2` | `results/lrcheck/k12s25n08b150_lr1e4_s3.json` |
| `k12s50n02b150_lr1e4_s2.json` | `2686569d21866b15` | `results/lrcheck/k12s50n02b150_lr1e4_s2.json` |
| `k12s50n04b100_lr1e4_s2.json` | `92c6e90b2d308950` | `results/lrcheck/k12s50n04b100_lr1e4_s2.json` |

51 files.

## `power/` — 4.2 (RQ2), figure rho_curve

RQ2, the answer-rate arm. Policies trained under a PARTIAL ORACLE -- ancestry answers withheld with probability 1 - rho, ~0.085 s/episode -- evaluated under genuine sampled evidence at 6-9 s/episode, which they never saw in training. The grid is 7 rates x 3 seeds x 200 paired episodes, taken from `deterministic/`: the same 21 cells were first built on an evaluation path that did not seed the torch RNG, so a learned arm scored with --sample was not reproducible, and `deterministic/` is the rebuild after that fix. Do not mix the two directories in one table. transfer_p{10,07,05} with p{10,07,05} is the isolation pair, differing in exactly one config field. repeat/ carries the coverage and repeat-rate mechanism probes; argmax/ the action-selection control.

Regenerate with: `scripts/run_rho_fleet.sh, then scripts/rebuild_grid_deterministic.sh, then scripts/rho_curve_report.py --dir results/power/rho/deterministic`

| file | sha256 (16) | source |
|---|---|---|
| `rho0.50_s0.json` | `ca356dcc6b230da5` | `results/power/rho/rho0.50_s0.json` |
| `rho0.50_s1.json` | `cb10d3d190c59167` | `results/power/rho/rho0.50_s1.json` |
| `rho0.50_s2.json` | `52ed21d8f5614c68` | `results/power/rho/rho0.50_s2.json` |
| `rho0.70_s0.json` | `88cdad49a6c71128` | `results/power/rho/rho0.70_s0.json` |
| `rho0.70_s1.json` | `2468a9e38af9e61c` | `results/power/rho/rho0.70_s1.json` |
| `rho0.70_s2.json` | `986ec7f3d0cb3b05` | `results/power/rho/rho0.70_s2.json` |
| `rho0.80_s0.json` | `60ec6406ec7944d4` | `results/power/rho/rho0.80_s0.json` |
| `rho0.80_s1.json` | `1e282ddaba1562d7` | `results/power/rho/rho0.80_s1.json` |
| `rho0.80_s2.json` | `14710e4b90832f4b` | `results/power/rho/rho0.80_s2.json` |
| `rho0.85_s0.json` | `897d65ba24a7291c` | `results/power/rho/rho0.85_s0.json` |
| `rho0.85_s1.json` | `284c68abeba10946` | `results/power/rho/rho0.85_s1.json` |
| `rho0.85_s2.json` | `8ccc4414a6e50dc7` | `results/power/rho/rho0.85_s2.json` |
| `rho0.90_s0.json` | `6bffbc0597d645c3` | `results/power/rho/rho0.90_s0.json` |
| `rho0.90_s1.json` | `fae22d360147cfbe` | `results/power/rho/rho0.90_s1.json` |
| `rho0.90_s2.json` | `a63deb40664eb522` | `results/power/rho/rho0.90_s2.json` |
| `rho0.95_long_s0.json` | `8289c20caf78372e` | `results/power/rho/rho0.95_long_s0.json` |
| `rho0.95_long_s1.json` | `78be49ca24e4a948` | `results/power/rho/rho0.95_long_s1.json` |
| `rho0.95_long_s2.json` | `66db49bd1d3de60b` | `results/power/rho/rho0.95_long_s2.json` |
| `rho0.95_s0.json` | `b982a980bbb551b8` | `results/power/rho/rho0.95_s0.json` |
| `rho0.95_s1.json` | `5cbbb20b41a07510` | `results/power/rho/rho0.95_s1.json` |
| `rho0.95_s2.json` | `2a5f25ac3336ca9e` | `results/power/rho/rho0.95_s2.json` |
| `rho1.00_s0.json` | `97d2ce2354d1ec81` | `results/power/rho/rho1.00_s0.json` |
| `rho1.00_s1.json` | `94cc7e902db40ca8` | `results/power/rho/rho1.00_s1.json` |
| `rho1.00_s2.json` | `9a1bb76b22a5b54e` | `results/power/rho/rho1.00_s2.json` |
| `CURVE.json` | `29b6b29a926498cd` | `results/power/rho/CURVE.json` |
| `transfer_p05.json` | `20562fd96445592b` | `results/power/transfer_p05.json` |
| `transfer_p07.json` | `24bb4fc9aae931af` | `results/power/transfer_p07.json` |
| `transfer_p10.json` | `acd0a97250b37eca` | `results/power/transfer_p10.json` |
| `p05.json` | `4cc44cc405408ed6` | `results/power/p05.json` |
| `p07.json` | `2baa6749b8f80914` | `results/power/p07.json` |
| `p10.json` | `b85bfd6e67662419` | `results/power/p10.json` |
| `p85.json` | `4dbd014a9b91facf` | `results/power/p85.json` |
| `p95.json` | `386735f2cd87eae5` | `results/power/p95.json` |
| `repeat_rho0.50.json` | `f2c954f0947ddbd0` | `results/power/rho/repeat/repeat_rho0.50.json` |
| `repeat_rho0.50_s1.json` | `814bab4d438b8204` | `results/power/rho/repeat/repeat_rho0.50_s1.json` |
| `repeat_rho0.50_s2.json` | `1819155db39fc840` | `results/power/rho/repeat/repeat_rho0.50_s2.json` |
| `repeat_rho0.70.json` | `1a76b9c6300f299a` | `results/power/rho/repeat/repeat_rho0.70.json` |
| `repeat_rho0.70_s1.json` | `f8d06e9f6958153e` | `results/power/rho/repeat/repeat_rho0.70_s1.json` |
| `repeat_rho0.70_s2.json` | `04e77a93576cf796` | `results/power/rho/repeat/repeat_rho0.70_s2.json` |
| `repeat_rho0.80.json` | `1931476dbf53a93c` | `results/power/rho/repeat/repeat_rho0.80.json` |
| `repeat_rho0.80_s1.json` | `283c30fa7bff2f0f` | `results/power/rho/repeat/repeat_rho0.80_s1.json` |
| `repeat_rho0.80_s2.json` | `ab90ad1413a4d02d` | `results/power/rho/repeat/repeat_rho0.80_s2.json` |
| `repeat_rho0.85.json` | `5e5253b1093ed5f8` | `results/power/rho/repeat/repeat_rho0.85.json` |
| `repeat_rho0.85_s1.json` | `aa6cfef4fd49cf04` | `results/power/rho/repeat/repeat_rho0.85_s1.json` |
| `repeat_rho0.85_s2.json` | `793d96761fb6f67f` | `results/power/rho/repeat/repeat_rho0.85_s2.json` |
| `repeat_rho0.90.json` | `e274f84748c65bc4` | `results/power/rho/repeat/repeat_rho0.90.json` |
| `repeat_rho0.90_s1.json` | `ece706c845378e2d` | `results/power/rho/repeat/repeat_rho0.90_s1.json` |
| `repeat_rho0.90_s2.json` | `bbf722e730cf3ad1` | `results/power/rho/repeat/repeat_rho0.90_s2.json` |
| `repeat_rho0.95.json` | `2996ba16afb0cac5` | `results/power/rho/repeat/repeat_rho0.95.json` |
| `repeat_rho0.95_s1.json` | `8a0d22c8e823385e` | `results/power/rho/repeat/repeat_rho0.95_s1.json` |
| `repeat_rho0.95_s2.json` | `2549ad8461c3ba8b` | `results/power/rho/repeat/repeat_rho0.95_s2.json` |
| `repeat_rho1.00.json` | `7dc626cdc8e63fed` | `results/power/rho/repeat/repeat_rho1.00.json` |
| `repeat_rho1.00_s1.json` | `fc70ae80e1b85cbc` | `results/power/rho/repeat/repeat_rho1.00_s1.json` |
| `repeat_rho1.00_s2.json` | `c0b3eebf3aedbbe6` | `results/power/rho/repeat/repeat_rho1.00_s2.json` |
| `argmax_rho0.70_s0.json` | `9d6645b775cc736b` | `results/power/rho/argmax/argmax_rho0.70_s0.json` |
| `argmax_rho0.70_s1.json` | `d8729d49959eb23e` | `results/power/rho/argmax/argmax_rho0.70_s1.json` |
| `argmax_rho0.70_s2.json` | `48ca40a80a5ea393` | `results/power/rho/argmax/argmax_rho0.70_s2.json` |
| `argmax_rho0.95_s0.json` | `63d30685fddc6d8c` | `results/power/rho/argmax/argmax_rho0.95_s0.json` |
| `argmax_rho0.95_s1.json` | `1108f5e1fd29191b` | `results/power/rho/argmax/argmax_rho0.95_s1.json` |
| `argmax_rho0.95_s2.json` | `9de55fd96017a624` | `results/power/rho/argmax/argmax_rho0.95_s2.json` |

60 files.
