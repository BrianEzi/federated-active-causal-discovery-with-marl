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
| `k04_best.json` | `a69472a8852b312d` | `results/ckpt/k04_best.json` |
| `k08_best.json` | `9752e542df815d18` | `results/ckpt/k08_best.json` |
| `k12_best.json` | `df2c5079070eacd6` | `results/ckpt/k12_best.json` |
| `k20_best.json` | `44a7d7e82f74814a` | `results/ckpt/k20_best.json` |
| `k30_best.json` | `78ba4536eac44d90` | `results/ckpt/k30_best.json` |
| `k04_final.json` | `76a35ffcc2d7be6b` | `results/ckpt/k04_final.json` |
| `k08_final.json` | `be09e3f4adb8236d` | `results/ckpt/k08_final.json` |
| `k12_final.json` | `03383c6688818d58` | `results/ckpt/k12_final.json` |
| `k20_final.json` | `a8d4c2abab76c318` | `results/ckpt/k20_final.json` |
| `k30_final.json` | `ca17233ca00d4e55` | `results/ckpt/k30_final.json` |

10 files.

## `attribution/` — 4.5 (RQ2), figure attribution_law

RQ2. attr_ceiling: recovery by group size and peer count. attr_ceiling_matched_budget: the control holding rounds-per-agent fixed, which is what rules out budget starvation. attr_ceiling_budget: the coverage step function (21/1056 at budget 30; 349/1056 at 60, 120 and 240 -- IDENTICAL counts, not merely equal rates). attr_scale_final and attr_reach: k=30/40/50 at 30 episodes each, zero misattributions. attr_train: training on the attribution reward. attr/transfer_*: the self-interested attribution baseline.

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

RQ3. v2_k12_* and v2_k20_*: arm A is the federated baseline, arm E removes the information partition (partners' beliefs and counts observed) and the optimiser partition (trajectories pooled instead of FedAvg). Action rights stay partitioned in both. shd_k20_*: the same arms on the primary metric, where the recovery rate has saturated and cannot separate them.

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
| `shd_A.json` | `e6767fc74a96188b` | `results/central/shd_A.json` |
| `shd_E.json` | `dae11cbc36ff37b3` | `results/central/shd_E.json` |
| `shd_k20_A.json` | `a953eb67e2405f70` | `results/central/shd_k20_A.json` |
| `shd_k20_E.json` | `b904860f7e0eb79e` | `results/central/shd_k20_E.json` |

25 files.
