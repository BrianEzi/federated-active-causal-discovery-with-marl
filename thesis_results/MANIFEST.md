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

## `sweep12k/` — RQ1, tables tab:12k_*, figure crossover_budget

THE PRIMARY SWEEP. Every cell retrained to 12,000 episodes, the budget at which the policies converge, and measured with scripts/global_shd_paired.py at the selected checkpoint, the final update, and update 500 (8,000 episodes). The 4,000-episode `sweep` folder is retained because Chapter 4 reports what that budget did to three structural claims, not because it is the headline. Joint recovery has the learned arm ahead of the myopic rule in 2 of 18 cells at 4,000 episodes and 16 of 18 at 12,000.

Regenerate with: `scripts/build_sweep12k.py to generate jobs, then scripts/measure_sweep12k.py --conventions best,final,u0500`

| file | sha256 (16) | source |
|---|---|---|
| `k04s50n04b150_s0.json` | `429e2434973f1d7f` | `results/sweep12k/k04s50n04b150_s0.json` |
| `k04s50n04b150_s1.json` | `43a7888ece40a830` | `results/sweep12k/k04s50n04b150_s1.json` |
| `k04s50n04b150_s2.json` | `7bd468b439ba5046` | `results/sweep12k/k04s50n04b150_s2.json` |
| `k08s50n04b150_s0.json` | `68d83035b67f8fd4` | `results/sweep12k/k08s50n04b150_s0.json` |
| `k08s50n04b150_s1.json` | `9b612ebf50129cc3` | `results/sweep12k/k08s50n04b150_s1.json` |
| `k08s50n04b150_s2.json` | `84985252a28a70db` | `results/sweep12k/k08s50n04b150_s2.json` |
| `k12s25n02b150_s0.json` | `cc1ba4529e676252` | `results/sweep12k/k12s25n02b150_s0.json` |
| `k12s25n02b150_s1.json` | `78c629a5f4e85438` | `results/sweep12k/k12s25n02b150_s1.json` |
| `k12s25n02b150_s2.json` | `1cf9d31735174b67` | `results/sweep12k/k12s25n02b150_s2.json` |
| `k12s25n04b150_s0.json` | `5ab20e4bc27ac30e` | `results/sweep12k/k12s25n04b150_s0.json` |
| `k12s25n04b150_s1.json` | `9096a658ca07c1f7` | `results/sweep12k/k12s25n04b150_s1.json` |
| `k12s25n04b150_s2.json` | `abc7c6b42e20421f` | `results/sweep12k/k12s25n04b150_s2.json` |
| `k12s25n08b150_s0.json` | `9aabc6d0e8acf230` | `results/sweep12k/k12s25n08b150_s0.json` |
| `k12s25n08b150_s1.json` | `6bf1460d2e98c860` | `results/sweep12k/k12s25n08b150_s1.json` |
| `k12s25n08b150_s2.json` | `3f80ee2d9aa2218d` | `results/sweep12k/k12s25n08b150_s2.json` |
| `k12s50n02b150_s0.json` | `bd2fa64602af5ef3` | `results/sweep12k/k12s50n02b150_s0.json` |
| `k12s50n02b150_s1.json` | `9add3212e496c01f` | `results/sweep12k/k12s50n02b150_s1.json` |
| `k12s50n02b150_s2.json` | `5047e1d62ad567d2` | `results/sweep12k/k12s50n02b150_s2.json` |
| `k12s50n03b150_s0.json` | `44b8336c5f1cb339` | `results/sweep12k/k12s50n03b150_s0.json` |
| `k12s50n03b150_s1.json` | `897d538f79fc708a` | `results/sweep12k/k12s50n03b150_s1.json` |
| `k12s50n03b150_s2.json` | `b28124e5fbdacaf6` | `results/sweep12k/k12s50n03b150_s2.json` |
| `k12s50n04b100_s0.json` | `4bc65994d1551985` | `results/sweep12k/k12s50n04b100_s0.json` |
| `k12s50n04b100_s1.json` | `d95fb0367e007ac0` | `results/sweep12k/k12s50n04b100_s1.json` |
| `k12s50n04b100_s2.json` | `f0136cc95df44ef5` | `results/sweep12k/k12s50n04b100_s2.json` |
| `k12s50n04b120_s0.json` | `90703f0a7313bbd8` | `results/sweep12k/k12s50n04b120_s0.json` |
| `k12s50n04b120_s1.json` | `2af007e59e8054a7` | `results/sweep12k/k12s50n04b120_s1.json` |
| `k12s50n04b120_s2.json` | `d609f059d8e68849` | `results/sweep12k/k12s50n04b120_s2.json` |
| `k12s50n04b150_s0.json` | `2848ae0da0436fe4` | `results/sweep12k/k12s50n04b150_s0.json` |
| `k12s50n04b150_s1.json` | `024677e453372a96` | `results/sweep12k/k12s50n04b150_s1.json` |
| `k12s50n04b150_s2.json` | `13eaef4f8212d4b4` | `results/sweep12k/k12s50n04b150_s2.json` |
| `k12s50n04b200_s0.json` | `4591e778f62a2607` | `results/sweep12k/k12s50n04b200_s0.json` |
| `k12s50n04b200_s1.json` | `7476179cdfd74ef1` | `results/sweep12k/k12s50n04b200_s1.json` |
| `k12s50n04b200_s2.json` | `c9795b82fe204435` | `results/sweep12k/k12s50n04b200_s2.json` |
| `k12s50n04b500_s0.json` | `9e9218e936f7db1a` | `results/sweep12k/k12s50n04b500_s0.json` |
| `k12s50n04b500_s1.json` | `cfebf9282e0a8a13` | `results/sweep12k/k12s50n04b500_s1.json` |
| `k12s50n04b500_s2.json` | `0f6e8dfe64e2fcc8` | `results/sweep12k/k12s50n04b500_s2.json` |
| `k12s50n05b150_s0.json` | `226aac33595dfb3e` | `results/sweep12k/k12s50n05b150_s0.json` |
| `k12s50n05b150_s1.json` | `9622e72de5ba93df` | `results/sweep12k/k12s50n05b150_s1.json` |
| `k12s50n05b150_s2.json` | `b8f7f73d1bf5eb24` | `results/sweep12k/k12s50n05b150_s2.json` |
| `k12s50n08b150_s0.json` | `0434b8f76e009ac6` | `results/sweep12k/k12s50n08b150_s0.json` |
| `k12s50n08b150_s1.json` | `a67e01770683e95b` | `results/sweep12k/k12s50n08b150_s1.json` |
| `k12s50n08b150_s2.json` | `159ee50fb9c70023` | `results/sweep12k/k12s50n08b150_s2.json` |
| `k12s50n10b150_s0.json` | `2a5a2a22d5f80e27` | `results/sweep12k/k12s50n10b150_s0.json` |
| `k12s50n10b150_s1.json` | `ac3f7b8895c4383c` | `results/sweep12k/k12s50n10b150_s1.json` |
| `k12s50n10b150_s2.json` | `599f2b8b882c10af` | `results/sweep12k/k12s50n10b150_s2.json` |
| `k12s75n02b150_s0.json` | `88ed4418e462b535` | `results/sweep12k/k12s75n02b150_s0.json` |
| `k12s75n02b150_s1.json` | `88a25cea0cdc36db` | `results/sweep12k/k12s75n02b150_s1.json` |
| `k12s75n02b150_s2.json` | `36fa7a3476fbb186` | `results/sweep12k/k12s75n02b150_s2.json` |
| `k12s75n04b150_s0.json` | `45a5f2528ee92438` | `results/sweep12k/k12s75n04b150_s0.json` |
| `k12s75n04b150_s1.json` | `273bc5ac56b35457` | `results/sweep12k/k12s75n04b150_s1.json` |
| `k12s75n04b150_s2.json` | `1041c0c4f3603aa3` | `results/sweep12k/k12s75n04b150_s2.json` |
| `k12s75n08b150_s0.json` | `cdb403e06218391f` | `results/sweep12k/k12s75n08b150_s0.json` |
| `k12s75n08b150_s1.json` | `b17a24e9914f185f` | `results/sweep12k/k12s75n08b150_s1.json` |
| `k12s75n08b150_s2.json` | `f071a3c55c6b714c` | `results/sweep12k/k12s75n08b150_s2.json` |
| `shd__k04s50n04b150.json` | `3d84d18829f42a84` | `results/sweep12k/shd/k04s50n04b150.json` |
| `shd__k08s50n04b150.json` | `308121a666a8cc58` | `results/sweep12k/shd/k08s50n04b150.json` |
| `shd__k12s25n02b150.json` | `63282c8f4f3ce801` | `results/sweep12k/shd/k12s25n02b150.json` |
| `shd__k12s25n04b150.json` | `0b2a60c97971d275` | `results/sweep12k/shd/k12s25n04b150.json` |
| `shd__k12s25n08b150.json` | `2d7524e70ff5866b` | `results/sweep12k/shd/k12s25n08b150.json` |
| `shd__k12s50n02b150.json` | `ee2d9d9635734826` | `results/sweep12k/shd/k12s50n02b150.json` |
| `shd__k12s50n03b150.json` | `014335b99f60cf0b` | `results/sweep12k/shd/k12s50n03b150.json` |
| `shd__k12s50n04b100.json` | `fbb46b7128cb37ac` | `results/sweep12k/shd/k12s50n04b100.json` |
| `shd__k12s50n04b120.json` | `eacc2053a6471ad0` | `results/sweep12k/shd/k12s50n04b120.json` |
| `shd__k12s50n04b150.json` | `92c6a9a3eaae41fd` | `results/sweep12k/shd/k12s50n04b150.json` |
| `shd__k12s50n04b200.json` | `a78d4a220681c6e2` | `results/sweep12k/shd/k12s50n04b200.json` |
| `shd__k12s50n04b500.json` | `f3efcebd5d03dd9d` | `results/sweep12k/shd/k12s50n04b500.json` |
| `shd__k12s75n02b150.json` | `56df3bc9b45ec427` | `results/sweep12k/shd/k12s75n02b150.json` |
| `shd__k12s75n08b150.json` | `c423f7a980194bdd` | `results/sweep12k/shd/k12s75n08b150.json` |
| `shd_final__k04s50n04b150.json` | `e818d6352fde6be2` | `results/sweep12k/shd_final/k04s50n04b150.json` |
| `shd_final__k08s50n04b150.json` | `d7d759da1e9b14e0` | `results/sweep12k/shd_final/k08s50n04b150.json` |
| `shd_final__k12s25n02b150.json` | `d3ca24d4d3ae6a83` | `results/sweep12k/shd_final/k12s25n02b150.json` |
| `shd_final__k12s25n04b150.json` | `b3b1451f92d1aada` | `results/sweep12k/shd_final/k12s25n04b150.json` |
| `shd_final__k12s25n08b150.json` | `289ff95762932a31` | `results/sweep12k/shd_final/k12s25n08b150.json` |
| `shd_final__k12s50n02b150.json` | `84a0ea037f3d51dc` | `results/sweep12k/shd_final/k12s50n02b150.json` |
| `shd_final__k12s50n03b150.json` | `7163af7f62e3387e` | `results/sweep12k/shd_final/k12s50n03b150.json` |
| `shd_final__k12s50n04b100.json` | `6880a6855d44babb` | `results/sweep12k/shd_final/k12s50n04b100.json` |
| `shd_final__k12s50n04b120.json` | `ec903fd7f9c2db72` | `results/sweep12k/shd_final/k12s50n04b120.json` |
| `shd_final__k12s50n04b150.json` | `7fa18c3509069f1e` | `results/sweep12k/shd_final/k12s50n04b150.json` |
| `shd_final__k12s50n04b200.json` | `5b0e70d7e8bd0e4c` | `results/sweep12k/shd_final/k12s50n04b200.json` |
| `shd_final__k12s50n04b500.json` | `544c8e9a3940cf0e` | `results/sweep12k/shd_final/k12s50n04b500.json` |
| `shd_final__k12s50n05b150.json` | `43a72825430126f7` | `results/sweep12k/shd_final/k12s50n05b150.json` |
| `shd_final__k12s50n08b150.json` | `e02e06957c21b868` | `results/sweep12k/shd_final/k12s50n08b150.json` |
| `shd_final__k12s50n10b150.json` | `70548377582aadb2` | `results/sweep12k/shd_final/k12s50n10b150.json` |
| `shd_final__k12s75n02b150.json` | `286682671e785991` | `results/sweep12k/shd_final/k12s75n02b150.json` |
| `shd_final__k12s75n04b150.json` | `08896e6c55488d97` | `results/sweep12k/shd_final/k12s75n04b150.json` |
| `shd_final__k12s75n08b150.json` | `c8e6a0e2ce230d0e` | `results/sweep12k/shd_final/k12s75n08b150.json` |
| `shd_u0500__k04s50n04b150.json` | `b513086afd14b95f` | `results/sweep12k/shd_u0500/k04s50n04b150.json` |
| `shd_u0500__k08s50n04b150.json` | `4a65425fa8b74da0` | `results/sweep12k/shd_u0500/k08s50n04b150.json` |
| `shd_u0500__k12s25n02b150.json` | `95daa1a36418adaf` | `results/sweep12k/shd_u0500/k12s25n02b150.json` |
| `shd_u0500__k12s25n04b150.json` | `9031880225dc8755` | `results/sweep12k/shd_u0500/k12s25n04b150.json` |
| `shd_u0500__k12s25n08b150.json` | `7d581b5671567680` | `results/sweep12k/shd_u0500/k12s25n08b150.json` |
| `shd_u0500__k12s50n02b150.json` | `39de3de2338f5ed4` | `results/sweep12k/shd_u0500/k12s50n02b150.json` |
| `shd_u0500__k12s50n03b150.json` | `f9ffa7632c3e313d` | `results/sweep12k/shd_u0500/k12s50n03b150.json` |
| `shd_u0500__k12s50n04b100.json` | `f2fc6aef1bb44bc5` | `results/sweep12k/shd_u0500/k12s50n04b100.json` |
| `shd_u0500__k12s50n04b120.json` | `e57edc9b6e787f43` | `results/sweep12k/shd_u0500/k12s50n04b120.json` |
| `shd_u0500__k12s50n04b150.json` | `8ec8225d4988890e` | `results/sweep12k/shd_u0500/k12s50n04b150.json` |
| `shd_u0500__k12s50n04b200.json` | `2ad5ec457eaa956b` | `results/sweep12k/shd_u0500/k12s50n04b200.json` |
| `shd_u0500__k12s50n04b500.json` | `e9ef89ba1eced374` | `results/sweep12k/shd_u0500/k12s50n04b500.json` |
| `shd_u0500__k12s50n05b150.json` | `8764eb151a982c5d` | `results/sweep12k/shd_u0500/k12s50n05b150.json` |
| `shd_u0500__k12s50n08b150.json` | `626c23dc5e696f2c` | `results/sweep12k/shd_u0500/k12s50n08b150.json` |
| `shd_u0500__k12s50n10b150.json` | `56b49aac8ba90d6e` | `results/sweep12k/shd_u0500/k12s50n10b150.json` |
| `shd_u0500__k12s75n02b150.json` | `a7bca6d023b69d49` | `results/sweep12k/shd_u0500/k12s75n02b150.json` |
| `shd_u0500__k12s75n04b150.json` | `85ee3518973509fa` | `results/sweep12k/shd_u0500/k12s75n04b150.json` |
| `shd_u0500__k12s75n08b150.json` | `2c19065afb8d90d6` | `results/sweep12k/shd_u0500/k12s75n08b150.json` |
| `shd_n05_12k.json` | `2765890352ab4825` | `results/longcheck/shd_n05_12k.json` |
| `shd_n08_12k.json` | `e226be006e03aede` | `results/longcheck/shd_n08_12k.json` |
| `shd_n10_12k.json` | `79711fe5ef3b1d09` | `results/longcheck/shd_n10_12k.json` |
| `shd_s75_12k.json` | `4fe3ca6da8ae32d2` | `results/longcheck/shd_s75_12k.json` |

108 files.

## `generator/` — RQ1, figure generator

The generator control: three Erdos-Renyi seeds at the principal cell, 12,000 episodes, current engine, measured at both conventions (identical). Densities near-matched to the scale-free comparator. The advantage holds on 3/3 seeds at 7-9 SE; the myopic rule degrades fifty-fold. Replaces results/vs_generator/ (superseded belief backend).

Regenerate with: `results/generator12k/run_generator12k.sh, then global_shd_paired.py at both conventions`

| file | sha256 (16) | source |
|---|---|---|
| `er_s0.json` | `601b0230090866d3` | `results/generator12k/er_s0.json` |
| `er_s1.json` | `aa927ec8758ca0e4` | `results/generator12k/er_s1.json` |
| `er_s2.json` | `36b0383e69c467f7` | `results/generator12k/er_s2.json` |
| `shd_er_best.json` | `233157ab440d9b97` | `results/generator12k/shd_er_best.json` |
| `shd_er_final.json` | `03989ec3bcde6a3e` | `results/generator12k/shd_er_final.json` |

5 files.

## `credit/` — RQ3, figure credit, tab:credit

Turn-aware credit at k=8 (and k=12 when the fill lands), measured with global_shd_paired.py. The recorded-field version showed a federation-specific 18x interaction that does not exist; measured, removing credit costs 15.1x pooled and 13.2x federated.

Regenerate with: `training runs in results/credit/, then scripts/global_shd_paired.py per 4-cell`

| file | sha256 (16) | source |
|---|---|---|
| `k08s50n04b150_E4_credit_s0.json` | `27f0198d8e7b22cb` | `results/credit/k08s50n04b150_E4_credit_s0.json` |
| `k08s50n04b150_E4_credit_s1.json` | `161fdbf176873c85` | `results/credit/k08s50n04b150_E4_credit_s1.json` |
| `k08s50n04b150_E4_credit_s2.json` | `ec8afeedfed90e18` | `results/credit/k08s50n04b150_E4_credit_s2.json` |
| `k08s50n04b150_E4_nocredit_s0.json` | `f297e7def76a2d61` | `results/credit/k08s50n04b150_E4_nocredit_s0.json` |
| `k08s50n04b150_E4_nocredit_s1.json` | `1137b444005a4982` | `results/credit/k08s50n04b150_E4_nocredit_s1.json` |
| `k08s50n04b150_E4_nocredit_s2.json` | `a58e14ed7fb9917a` | `results/credit/k08s50n04b150_E4_nocredit_s2.json` |
| `k08s50n04b150_pooled_credit_s0.json` | `e1e474af234394d2` | `results/credit/k08s50n04b150_pooled_credit_s0.json` |
| `k08s50n04b150_pooled_credit_s1.json` | `4cecb274d1be5f0b` | `results/credit/k08s50n04b150_pooled_credit_s1.json` |
| `k08s50n04b150_pooled_credit_s2.json` | `bc5c0546d4bf279c` | `results/credit/k08s50n04b150_pooled_credit_s2.json` |
| `k08s50n04b150_pooled_nocredit_s0.json` | `edaf934d7ffdc9fc` | `results/credit/k08s50n04b150_pooled_nocredit_s0.json` |
| `k08s50n04b150_pooled_nocredit_s1.json` | `2189dacb33dd795f` | `results/credit/k08s50n04b150_pooled_nocredit_s1.json` |
| `k08s50n04b150_pooled_nocredit_s2.json` | `d4b7dcc720549b4a` | `results/credit/k08s50n04b150_pooled_nocredit_s2.json` |
| `k12s50n04b150_E4_credit_s0.json` | `5829a45e2d36e0b9` | `results/credit/k12s50n04b150_E4_credit_s0.json` |
| `k12s50n04b150_E4_credit_s1.json` | `48102b34b595191d` | `results/credit/k12s50n04b150_E4_credit_s1.json` |
| `k12s50n04b150_E4_credit_s2.json` | `168424e77567a055` | `results/credit/k12s50n04b150_E4_credit_s2.json` |
| `k12s50n04b150_E4_nocredit_s0.json` | `6011972ab8c33c06` | `results/credit/k12s50n04b150_E4_nocredit_s0.json` |
| `k12s50n04b150_E4_nocredit_s1.json` | `8766af8a62d3b044` | `results/credit/k12s50n04b150_E4_nocredit_s1.json` |
| `k12s50n04b150_E4_nocredit_s2.json` | `8df0bffb73cbbfd4` | `results/credit/k12s50n04b150_E4_nocredit_s2.json` |
| `k12s50n04b150_pooled_credit_s0.json` | `5dc2820676fd950b` | `results/credit/k12s50n04b150_pooled_credit_s0.json` |
| `k12s50n04b150_pooled_credit_s1.json` | `ec6b1653b3f29ff3` | `results/credit/k12s50n04b150_pooled_credit_s1.json` |
| `k12s50n04b150_pooled_credit_s2.json` | `795147e5835d986b` | `results/credit/k12s50n04b150_pooled_credit_s2.json` |
| `k12s50n04b150_pooled_nocredit_s0.json` | `bcbd9e7eb0294787` | `results/credit/k12s50n04b150_pooled_nocredit_s0.json` |
| `k12s50n04b150_pooled_nocredit_s1.json` | `57eb0b477ad4f8bb` | `results/credit/k12s50n04b150_pooled_nocredit_s1.json` |
| `k12s50n04b150_pooled_nocredit_s2.json` | `8a27c9977f0e1602` | `results/credit/k12s50n04b150_pooled_nocredit_s2.json` |
| `k08s50n04b150_E4_credit.json` | `4a0433af3368dcd8` | `results/credit/shd/k08s50n04b150_E4_credit.json` |
| `k08s50n04b150_E4_nocredit.json` | `a305b9e35e08fa4a` | `results/credit/shd/k08s50n04b150_E4_nocredit.json` |
| `k08s50n04b150_pooled_credit.json` | `9337cdd7d54785f9` | `results/credit/shd/k08s50n04b150_pooled_credit.json` |
| `k08s50n04b150_pooled_nocredit.json` | `8bfa6b32e424f3a0` | `results/credit/shd/k08s50n04b150_pooled_nocredit.json` |
| `k12s50n04b150_E4_credit.json` | `f1e43674a531c3b0` | `results/credit/shd/k12s50n04b150_E4_credit.json` |
| `k12s50n04b150_E4_nocredit.json` | `bf866220c8ee663e` | `results/credit/shd/k12s50n04b150_E4_nocredit.json` |
| `k12s50n04b150_pooled_credit.json` | `287fbc423b93b390` | `results/credit/shd/k12s50n04b150_pooled_credit.json` |
| `k12s50n04b150_pooled_nocredit.json` | `f8bba6baa402d24d` | `results/credit/shd/k12s50n04b150_pooled_nocredit.json` |

32 files.

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
| `k20_u0249.json` | `334b4ab147ccd716` | `results/rerows/k20_u0249.json` |
| `k30_u0249.json` | `3a0cb52e305f5146` | `results/rerows/k30_u0249.json` |

12 files.

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
| `v2_k12_A_s0.json` | `410f243de984d341` | `results/central12k/v2_k12_A_s0.json` |
| `v2_k12_A_s1.json` | `8db90530a1ccd274` | `results/central12k/v2_k12_A_s1.json` |
| `v2_k12_A_s2.json` | `a03a46259f83bf3c` | `results/central12k/v2_k12_A_s2.json` |
| `v2_k12_A_s3.json` | `3833f3921952497b` | `results/central12k/v2_k12_A_s3.json` |
| `v2_k12_A_s4.json` | `767f2439aab52aa0` | `results/central12k/v2_k12_A_s4.json` |
| `v2_k12_A_s5.json` | `c98dcfc6c4e68cea` | `results/central12k/v2_k12_A_s5.json` |
| `v2_k12_E_s0.json` | `7e600771d8f54fb8` | `results/central12k/v2_k12_E_s0.json` |
| `v2_k12_E_s1.json` | `1bca76da074a11ea` | `results/central12k/v2_k12_E_s1.json` |
| `v2_k12_E_s2.json` | `5991603b58965e2c` | `results/central12k/v2_k12_E_s2.json` |
| `v2_k12_E_s3.json` | `a77a94f32e405f7e` | `results/central12k/v2_k12_E_s3.json` |
| `v2_k12_E_s4.json` | `f7c7c60eae6b0100` | `results/central12k/v2_k12_E_s4.json` |
| `v2_k12_E_s5.json` | `3ad79ba23b2a616d` | `results/central12k/v2_k12_E_s5.json` |
| `v2_k20_A_s0.json` | `90a4228cdbe9a8ee` | `results/central/v2_k20_A_s0.json` |
| `v2_k20_A_s1.json` | `4cd4ccd0a0de7d5b` | `results/central/v2_k20_A_s1.json` |
| `v2_k20_A_s2.json` | `c0505184b28b21ae` | `results/central/v2_k20_A_s2.json` |
| `v2_k20_E_s0.json` | `da8fee69af83ca4a` | `results/central/v2_k20_E_s0.json` |
| `v2_k20_E_s1.json` | `49c26918edd99849` | `results/central/v2_k20_E_s1.json` |
| `v2_k20_E_s2.json` | `fa486406b54162fd` | `results/central/v2_k20_E_s2.json` |
| `ladder12k_A_best.json` | `56692c1baf4bc49d` | `results/rerows/ladder12k_A_best.json` |
| `ladder12k_E_best.json` | `d0cc6817478da850` | `results/rerows/ladder12k_E_best.json` |
| `ladder12k_A_final.json` | `e870b110a3eef2f6` | `results/rerows/ladder12k_A_final.json` |
| `ladder12k_E_final.json` | `889ea06c98a25858` | `results/rerows/ladder12k_E_final.json` |
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

52 files.

## `power/` — 4.2 (RQ2), figure rho_curve

RQ2, the answer-rate arm. Policies trained under a PARTIAL ORACLE -- ancestry answers withheld with probability 1 - rho, ~0.085 s/episode -- evaluated under genuine sampled evidence at 6-9 s/episode, which they never saw in training. The grid is 7 rates x 3 seeds x 200 paired episodes, taken from `deterministic/`: the same 21 cells were first built on an evaluation path that did not seed the torch RNG, so a learned arm scored with --sample was not reproducible, and `deterministic/` is the rebuild after that fix. Do not mix the two directories in one table. transfer_p{10,07,05} with p{10,07,05} is the isolation pair, differing in exactly one config field. repeat/ carries the coverage and repeat-rate mechanism probes; argmax/ the action-selection control.

Regenerate with: `scripts/run_rho_fleet.sh, then scripts/rebuild_grid_deterministic.sh, then scripts/rho_curve_report.py --dir results/power/rho/deterministic`

| file | sha256 (16) | source |
|---|---|---|
| `xfer_rho0.50_s0.json` | `0ba0500f6deb4b18` | `results/power/rho/deterministic/xfer_rho0.50_s0.json` |
| `xfer_rho0.50_s1.json` | `d575add3c8d60fef` | `results/power/rho/deterministic/xfer_rho0.50_s1.json` |
| `xfer_rho0.50_s2.json` | `e41dd38155c5938a` | `results/power/rho/deterministic/xfer_rho0.50_s2.json` |
| `xfer_rho0.70_s0.json` | `d3fc2cbcca241323` | `results/power/rho/deterministic/xfer_rho0.70_s0.json` |
| `xfer_rho0.70_s1.json` | `c63754e31695c639` | `results/power/rho/deterministic/xfer_rho0.70_s1.json` |
| `xfer_rho0.70_s2.json` | `31af894a0c9e8ea0` | `results/power/rho/deterministic/xfer_rho0.70_s2.json` |
| `xfer_rho0.80_s0.json` | `ac2b29d033dce334` | `results/power/rho/deterministic/xfer_rho0.80_s0.json` |
| `xfer_rho0.80_s1.json` | `29a16b2bc164eaad` | `results/power/rho/deterministic/xfer_rho0.80_s1.json` |
| `xfer_rho0.80_s2.json` | `8fe06e4701f0fe84` | `results/power/rho/deterministic/xfer_rho0.80_s2.json` |
| `xfer_rho0.85_s0.json` | `c69763650ca03621` | `results/power/rho/deterministic/xfer_rho0.85_s0.json` |
| `xfer_rho0.85_s1.json` | `c969cf44f3188c20` | `results/power/rho/deterministic/xfer_rho0.85_s1.json` |
| `xfer_rho0.85_s2.json` | `92172c30e0341442` | `results/power/rho/deterministic/xfer_rho0.85_s2.json` |
| `xfer_rho0.90_s0.json` | `300c783d6031db77` | `results/power/rho/deterministic/xfer_rho0.90_s0.json` |
| `xfer_rho0.90_s1.json` | `973099139554249e` | `results/power/rho/deterministic/xfer_rho0.90_s1.json` |
| `xfer_rho0.90_s2.json` | `b3c5dc3e85af6a12` | `results/power/rho/deterministic/xfer_rho0.90_s2.json` |
| `xfer_rho0.95_s0.json` | `88347059de38df70` | `results/power/rho/deterministic/xfer_rho0.95_s0.json` |
| `xfer_rho0.95_s1.json` | `d87b526ac75563c4` | `results/power/rho/deterministic/xfer_rho0.95_s1.json` |
| `xfer_rho0.95_s2.json` | `9fd2932f19438c06` | `results/power/rho/deterministic/xfer_rho0.95_s2.json` |
| `xfer_rho1.00_s0.json` | `3a20c6ab520421a5` | `results/power/rho/deterministic/xfer_rho1.00_s0.json` |
| `xfer_rho1.00_s1.json` | `4e9e089e7cf6e685` | `results/power/rho/deterministic/xfer_rho1.00_s1.json` |
| `xfer_rho1.00_s2.json` | `8c10d0da829755f2` | `results/power/rho/deterministic/xfer_rho1.00_s2.json` |
| `rho__rho0.50_s0.json` | `ca356dcc6b230da5` | `results/power/rho/rho0.50_s0.json` |
| `rho__rho0.50_s1.json` | `cb10d3d190c59167` | `results/power/rho/rho0.50_s1.json` |
| `rho__rho0.50_s2.json` | `52ed21d8f5614c68` | `results/power/rho/rho0.50_s2.json` |
| `rho__rho0.70_s0.json` | `88cdad49a6c71128` | `results/power/rho/rho0.70_s0.json` |
| `rho__rho0.70_s1.json` | `2468a9e38af9e61c` | `results/power/rho/rho0.70_s1.json` |
| `rho__rho0.70_s2.json` | `986ec7f3d0cb3b05` | `results/power/rho/rho0.70_s2.json` |
| `rho__rho0.80_s0.json` | `60ec6406ec7944d4` | `results/power/rho/rho0.80_s0.json` |
| `rho__rho0.80_s1.json` | `1e282ddaba1562d7` | `results/power/rho/rho0.80_s1.json` |
| `rho__rho0.80_s2.json` | `14710e4b90832f4b` | `results/power/rho/rho0.80_s2.json` |
| `rho__rho0.85_s0.json` | `897d65ba24a7291c` | `results/power/rho/rho0.85_s0.json` |
| `rho__rho0.85_s1.json` | `284c68abeba10946` | `results/power/rho/rho0.85_s1.json` |
| `rho__rho0.85_s2.json` | `8ccc4414a6e50dc7` | `results/power/rho/rho0.85_s2.json` |
| `rho__rho0.90_s0.json` | `6bffbc0597d645c3` | `results/power/rho/rho0.90_s0.json` |
| `rho__rho0.90_s1.json` | `fae22d360147cfbe` | `results/power/rho/rho0.90_s1.json` |
| `rho__rho0.90_s2.json` | `a63deb40664eb522` | `results/power/rho/rho0.90_s2.json` |
| `rho0.95_long_s0.json` | `8289c20caf78372e` | `results/power/rho/rho0.95_long_s0.json` |
| `rho0.95_long_s1.json` | `78be49ca24e4a948` | `results/power/rho/rho0.95_long_s1.json` |
| `rho0.95_long_s2.json` | `66db49bd1d3de60b` | `results/power/rho/rho0.95_long_s2.json` |
| `rho__rho0.95_s0.json` | `b982a980bbb551b8` | `results/power/rho/rho0.95_s0.json` |
| `rho__rho0.95_s1.json` | `5cbbb20b41a07510` | `results/power/rho/rho0.95_s1.json` |
| `rho__rho0.95_s2.json` | `2a5f25ac3336ca9e` | `results/power/rho/rho0.95_s2.json` |
| `rho__rho1.00_s0.json` | `97d2ce2354d1ec81` | `results/power/rho/rho1.00_s0.json` |
| `rho__rho1.00_s1.json` | `94cc7e902db40ca8` | `results/power/rho/rho1.00_s1.json` |
| `rho__rho1.00_s2.json` | `9a1bb76b22a5b54e` | `results/power/rho/rho1.00_s2.json` |
| `CURVE.json` | `33f53c9c2a7f0571` | `results/power/rho/CURVE.json` |
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
| `inregime_det__rho0.50_s0.json` | `6d419f9f942610cf` | `results/power/rho/inregime_det/rho0.50_s0.json` |
| `inregime_det__rho0.50_s1.json` | `0e04cf15ee9b3b73` | `results/power/rho/inregime_det/rho0.50_s1.json` |
| `inregime_det__rho0.50_s2.json` | `6aa25b83b3dcea0a` | `results/power/rho/inregime_det/rho0.50_s2.json` |
| `inregime_det__rho0.70_s0.json` | `d442c159cbd74a5d` | `results/power/rho/inregime_det/rho0.70_s0.json` |
| `inregime_det__rho0.70_s1.json` | `15fd3c9634693c4b` | `results/power/rho/inregime_det/rho0.70_s1.json` |
| `inregime_det__rho0.70_s2.json` | `e9dde5226b1aa17c` | `results/power/rho/inregime_det/rho0.70_s2.json` |
| `inregime_det__rho0.80_s0.json` | `bd3e646cfafcafc7` | `results/power/rho/inregime_det/rho0.80_s0.json` |
| `inregime_det__rho0.80_s1.json` | `615d6813730d651a` | `results/power/rho/inregime_det/rho0.80_s1.json` |
| `inregime_det__rho0.80_s2.json` | `3491eb2aa3477c5c` | `results/power/rho/inregime_det/rho0.80_s2.json` |
| `inregime_det__rho0.85_s0.json` | `c324dea380fac143` | `results/power/rho/inregime_det/rho0.85_s0.json` |
| `inregime_det__rho0.85_s1.json` | `1a751096b516670b` | `results/power/rho/inregime_det/rho0.85_s1.json` |
| `inregime_det__rho0.85_s2.json` | `dfdaee602e9b27ee` | `results/power/rho/inregime_det/rho0.85_s2.json` |
| `inregime_det__rho0.90_s0.json` | `d96f1ef2fe71b28d` | `results/power/rho/inregime_det/rho0.90_s0.json` |
| `inregime_det__rho0.90_s1.json` | `970413ee1dcd046c` | `results/power/rho/inregime_det/rho0.90_s1.json` |
| `inregime_det__rho0.90_s2.json` | `5b68bbf32dd483fc` | `results/power/rho/inregime_det/rho0.90_s2.json` |
| `inregime_det__rho0.95_s0.json` | `8462c975935707aa` | `results/power/rho/inregime_det/rho0.95_s0.json` |
| `inregime_det__rho0.95_s1.json` | `28aff7d6018b7aa3` | `results/power/rho/inregime_det/rho0.95_s1.json` |
| `inregime_det__rho0.95_s2.json` | `eaa3352a77a68a90` | `results/power/rho/inregime_det/rho0.95_s2.json` |
| `inregime_det__rho1.00_s0.json` | `a4a21aa47e685b75` | `results/power/rho/inregime_det/rho1.00_s0.json` |
| `inregime_det__rho1.00_s1.json` | `49f359d14ead04a0` | `results/power/rho/inregime_det/rho1.00_s1.json` |
| `inregime_det__rho1.00_s2.json` | `fab91ea24cc72235` | `results/power/rho/inregime_det/rho1.00_s2.json` |
| `fixed_rho0.50_s0_evalp0.5.json` | `6d419f9f942610cf` | `results/power/rho/evalsweep_det/fixed_rho0.50_s0_evalp0.5.json` |
| `fixed_rho0.50_s0_evalp0.7.json` | `893518112fcc2462` | `results/power/rho/evalsweep_det/fixed_rho0.50_s0_evalp0.7.json` |
| `fixed_rho0.50_s0_evalp0.8.json` | `f97168e261442f4a` | `results/power/rho/evalsweep_det/fixed_rho0.50_s0_evalp0.8.json` |
| `fixed_rho0.50_s0_evalp0.9.json` | `fa3a5b9a6ca14b9c` | `results/power/rho/evalsweep_det/fixed_rho0.50_s0_evalp0.9.json` |
| `fixed_rho0.50_s0_evalp1.0.json` | `e0f2f1e074929882` | `results/power/rho/evalsweep_det/fixed_rho0.50_s0_evalp1.0.json` |
| `fixed_rho0.50_s1_evalp0.5.json` | `0e04cf15ee9b3b73` | `results/power/rho/evalsweep_det/fixed_rho0.50_s1_evalp0.5.json` |
| `fixed_rho0.50_s1_evalp0.7.json` | `c3df0b7bb08ff81e` | `results/power/rho/evalsweep_det/fixed_rho0.50_s1_evalp0.7.json` |
| `fixed_rho0.50_s1_evalp0.8.json` | `0ce80b4bfe5e7d71` | `results/power/rho/evalsweep_det/fixed_rho0.50_s1_evalp0.8.json` |
| `fixed_rho0.50_s1_evalp0.9.json` | `58444f2595a18a92` | `results/power/rho/evalsweep_det/fixed_rho0.50_s1_evalp0.9.json` |
| `fixed_rho0.50_s1_evalp1.0.json` | `1c1e196de9b7c9eb` | `results/power/rho/evalsweep_det/fixed_rho0.50_s1_evalp1.0.json` |
| `fixed_rho0.50_s2_evalp0.5.json` | `6aa25b83b3dcea0a` | `results/power/rho/evalsweep_det/fixed_rho0.50_s2_evalp0.5.json` |
| `fixed_rho0.50_s2_evalp0.7.json` | `b855001e9c6c7662` | `results/power/rho/evalsweep_det/fixed_rho0.50_s2_evalp0.7.json` |
| `fixed_rho0.50_s2_evalp0.8.json` | `a2bfb8a9896c1727` | `results/power/rho/evalsweep_det/fixed_rho0.50_s2_evalp0.8.json` |
| `fixed_rho0.50_s2_evalp0.9.json` | `4e40e18980621d4a` | `results/power/rho/evalsweep_det/fixed_rho0.50_s2_evalp0.9.json` |
| `fixed_rho0.50_s2_evalp1.0.json` | `6506a2c0a82f8b41` | `results/power/rho/evalsweep_det/fixed_rho0.50_s2_evalp1.0.json` |
| `fixed_rho1.00_s0_evalp0.5.json` | `d4025212f10414c1` | `results/power/rho/evalsweep_det/fixed_rho1.00_s0_evalp0.5.json` |
| `fixed_rho1.00_s0_evalp0.7.json` | `4add85d7e0c27456` | `results/power/rho/evalsweep_det/fixed_rho1.00_s0_evalp0.7.json` |
| `fixed_rho1.00_s0_evalp0.8.json` | `7b52c443bfab65ae` | `results/power/rho/evalsweep_det/fixed_rho1.00_s0_evalp0.8.json` |
| `fixed_rho1.00_s0_evalp0.9.json` | `9e0b60dc739d993d` | `results/power/rho/evalsweep_det/fixed_rho1.00_s0_evalp0.9.json` |
| `fixed_rho1.00_s0_evalp1.0.json` | `a4a21aa47e685b75` | `results/power/rho/evalsweep_det/fixed_rho1.00_s0_evalp1.0.json` |
| `fixed_rho1.00_s1_evalp0.5.json` | `870ab35b45cc05b3` | `results/power/rho/evalsweep_det/fixed_rho1.00_s1_evalp0.5.json` |
| `fixed_rho1.00_s1_evalp0.7.json` | `c795af0d798eb223` | `results/power/rho/evalsweep_det/fixed_rho1.00_s1_evalp0.7.json` |
| `fixed_rho1.00_s1_evalp0.8.json` | `74d7e54330e1fc80` | `results/power/rho/evalsweep_det/fixed_rho1.00_s1_evalp0.8.json` |
| `fixed_rho1.00_s1_evalp0.9.json` | `d567eeb2deb7e398` | `results/power/rho/evalsweep_det/fixed_rho1.00_s1_evalp0.9.json` |
| `fixed_rho1.00_s1_evalp1.0.json` | `49f359d14ead04a0` | `results/power/rho/evalsweep_det/fixed_rho1.00_s1_evalp1.0.json` |
| `fixed_rho1.00_s2_evalp0.5.json` | `e00c3514660eab45` | `results/power/rho/evalsweep_det/fixed_rho1.00_s2_evalp0.5.json` |
| `fixed_rho1.00_s2_evalp0.7.json` | `3ae327328f24c1b3` | `results/power/rho/evalsweep_det/fixed_rho1.00_s2_evalp0.7.json` |
| `fixed_rho1.00_s2_evalp0.8.json` | `257bc6c357462d12` | `results/power/rho/evalsweep_det/fixed_rho1.00_s2_evalp0.8.json` |
| `fixed_rho1.00_s2_evalp0.9.json` | `0296a04aaeef06ca` | `results/power/rho/evalsweep_det/fixed_rho1.00_s2_evalp0.9.json` |
| `fixed_rho1.00_s2_evalp1.0.json` | `fab91ea24cc72235` | `results/power/rho/evalsweep_det/fixed_rho1.00_s2_evalp1.0.json` |
| `nint200.json` | `5e6f100544385577` | `results/sampled_det/nint200.json` |

127 files.
