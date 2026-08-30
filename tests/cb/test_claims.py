"""Three-outcome claim scoring, the claims reward, episode mix, stratified resampling,
and the constraint-side greedy -- the Day-1 redesign, each piece checked directly."""
from __future__ import annotations

import numpy as np
import pytest

from cb.bootstrap import BootstrapBelief, bootstrap_belief
from cb.claims import score_window
from ma.baselines import UncertaintyGreedyAgent
from ma.env import VARY, MAConfig, TwoAgentEnv, ROUND_ROBIN
from ma.projection import BIDIRECTED as MB, DIRECTED as MD, bidirected_pairs
from ma.topology import two_agent

TOPO = two_agent(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))


def _belief(adjacency, directed, bidirected):
    k = np.asarray(adjacency).shape[0]
    return BootstrapBelief(np.asarray(directed, float), np.asarray(bidirected, float),
                           np.asarray(adjacency, float), n_boot=12, ci_tests=0,
                           truncated_fraction=0.0)


def _mag_chain_conf():
    """0 -> 1 directed, (1, 2) confounded."""
    mag = np.zeros((3, 3), dtype=np.int8)
    mag[0, 1] = MD
    mag[1, 2] = mag[2, 1] = MB
    return mag


def test_perfect_belief_scores_all_right_and_identifies():
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    directed = np.zeros((3, 3)); directed[0, 1] = 1.0
    bidirected = np.zeros((3, 3)); bidirected[1, 2] = bidirected[2, 1] = 1.0
    s = score_window(_belief(adjacency, directed, bidirected), _mag_chain_conf(),
                     private_positions=[0])
    # 3 adjacency claims + 2 type claims, all right; all 5 required (0 is private).
    assert (s.n_right, s.n_wrong, s.n_unsure) == (5, 0, 0)
    assert s.identified and s.fraction() == 1.0


def test_unsure_is_neither_right_nor_wrong_and_blocks_identification():
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    directed = np.zeros((3, 3)); directed[0, 1] = 0.5          # below the 0.7 bar
    bidirected = np.zeros((3, 3)); bidirected[1, 2] = bidirected[2, 1] = 1.0
    s = score_window(_belief(adjacency, directed, bidirected), _mag_chain_conf(),
                     private_positions=[0])
    assert (s.n_right, s.n_wrong, s.n_unsure) == (4, 0, 1)
    assert not s.identified                                    # required claim unsure
    assert 0.0 < s.fraction() < 1.0


def test_settled_wrong_is_punished_and_vetoes_identification():
    """A confident wrong answer must cost MORE than an open question, and even a
    non-required wrong claim vetoes identification -- confidently wrong anywhere is
    not identified."""
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    directed = np.zeros((3, 3)); directed[0, 1] = 1.0
    bidirected = np.zeros((3, 3))
    directed[1, 2] = 1.0        # true confounding called DIRECTED: settled-wrong
    s = score_window(_belief(adjacency, directed, bidirected), _mag_chain_conf())
    assert s.n_wrong == 1
    unsure_version = score_window(
        _belief(adjacency, directed * [[0, 1, 0], [0, 0, 0.5], [0, 0, 0]], bidirected),
        _mag_chain_conf())
    assert s.fraction() < unsure_version.fraction()
    assert not s.identified


def test_every_type_claim_is_required_by_default():
    """A shared-block direction left unsure BLOCKS identification (2026-08-26).

    This test previously asserted the opposite, on the ground that "Markov equivalence
    leaves such edges unorientable". That was wrong: Markov equivalence constrains
    OBSERVATION, and the interventional reveal channel is pairwise ancestry, so
    intervening on both endpoints of an adjacent pair fixes its mark outright. The
    exemption was a grading choice, and it exempted precisely the contended surface that
    coordination exists to divide. `require_all_types=False` keeps the old grading
    available for reproducing pre-2026-08-26 numbers.
    """
    mag = np.zeros((3, 3), dtype=np.int8)
    mag[1, 2] = MD                                     # edge between two non-private
    adjacency = np.zeros((3, 3)); adjacency[1, 2] = adjacency[2, 1] = 1.0
    directed = np.zeros((3, 3)); directed[1, 2] = 0.4  # unsure
    belief = _belief(adjacency, directed, np.zeros((3, 3)))

    strict = score_window(belief, mag, private_positions=[0])
    assert strict.n_unsure == 1
    assert strict.required_total == 4                  # 3 adjacency + 1 type
    assert not strict.identified

    legacy = score_window(belief, mag, private_positions=[0], require_all_types=False)
    assert legacy.required_total == 3                  # the type claim is exempt
    assert legacy.identified


# -- episode mix --------------------------------------------------------------------


def _env(**kw):
    kw.setdefault("belief_backend", "constraint")
    kw.setdefault("cb_n_boot", 4)
    kw.setdefault("action_modes", (VARY,))
    return TwoAgentEnv(MAConfig(topology=TOPO, n_obs=120, n_int=40, budget=3,
                                turn_order=ROUND_ROBIN, **kw), seed=0)


def _is_confounded(env):
    return any(bidirected_pairs(env.true_adjacency, tuple(w.nodes))
               for w in env.windows.values())


def test_episode_mix_controls_confounding():
    conf = _env(episode_mix="confounded")
    clean = _env(episode_mix="unconfounded")
    for seed in range(6):
        conf.reset(seed=seed)
        assert _is_confounded(conf)
        clean.reset(seed=seed)
        assert not _is_confounded(clean)


def test_mix_draws_is_reported():
    env = _env(episode_mix="confounded")
    result = env.reset(seed=1)
    assert result.info["mix_draws"] >= 1


# -- the claims reward through the env -------------------------------------------------


def test_claims_reward_runs_and_reports():
    env = _env(reward_criterion="claims", episode_mix="confounded")
    result = env.reset(seed=2)
    assert result.info["claims"] is not None
    total = {a: sum(result.info["claims"][a][key] for key in ("right", "wrong", "unsure"))
             for a in (0, 1)}
    assert all(t > 0 for t in total.values())
    while not result.done:
        acts = {a: 0 for a in (0, 1)}
        result = env.step(acts)
    assert np.isfinite(result.reward)


def test_claims_reward_refused_on_exact_backend():
    with pytest.raises(ValueError, match="claims"):
        TwoAgentEnv(MAConfig(topology=TOPO, reward_criterion="claims"), seed=0)


# -- stratified resampling ---------------------------------------------------------------


def test_every_row_its_own_block_makes_resampling_the_identity():
    """The sharpest property test available: one row per block means the stratified
    resample must reproduce the dataset exactly, so every replicate equals the first."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 3))
    data[:, 2] = data[:, 0] + rng.normal(size=200) * 0.3
    belief = bootstrap_belief(data, n_boot=8, blocks=np.arange(200))
    for codes in belief.replicates[1:]:
        assert np.array_equal(codes, belief.replicates[0])


# -- the constraint-side greedy ---------------------------------------------------------


def test_uncertainty_greedy_targets_the_open_question_and_passes_when_done():
    env = _env()
    env.reset(seed=3)
    agent = UncertaintyGreedyAgent(0, seed=0)
    window = env.windows[0]
    k = window.k

    settled_adj = np.zeros((k, k)); settled_dir = np.zeros((k, k))
    settled_adj[0, 1] = settled_adj[1, 0] = 1.0
    settled_dir[0, 1] = 1.0
    unsure = BootstrapBelief(settled_dir, np.zeros((k, k)), settled_adj,
                             n_boot=12, ci_tests=0, truncated_fraction=0.0)
    unsure.adjacency[2, 3] = unsure.adjacency[3, 2] = 0.6      # open adjacency question
    window.belief.last = unsure
    action = agent(env, None)
    node, _ = window.actions[action]
    assert window.pos[node] in (2, 3), "greedy must target a node touching the unsure claim"

    unsure.adjacency[2, 3] = unsure.adjacency[3, 2] = 0.0      # now everything settled
    assert agent(env, None) == window.pass_index


# -- the oracle warm start ---------------------------------------------------------------


def test_oracle_skeleton_is_the_observational_limit_and_leaks_nothing():
    """Hidden 0 -> {1, 2}: the pair (1, 2) must start ADJACENT (observation cannot
    explain it away) but NOT confounded -- detecting that stays the interventions' job."""
    import numpy as np
    from ma.projection import observational_skeleton
    adjacency = np.zeros((4, 4), dtype=int)
    adjacency[0, 1] = adjacency[0, 2] = adjacency[2, 3] = 1
    adj, sepsets = observational_skeleton(adjacency, (1, 2, 3))
    # window positions: 0=node1, 1=node2, 2=node3
    assert adj[0, 1], "confounded pair must remain adjacent"
    assert adj[1, 2], "real edge 2->3 must be adjacent"
    assert not adj[0, 2] and (0, 2) in sepsets, "1 and 3 separate (via node 2)"


def test_oracle_warm_start_settles_adjacency_and_only_adjacency():
    env = _env(oracle_obs_structure=True, reward_criterion="claims",
               episode_mix="confounded")
    result = env.reset(seed=4)
    for agent, window in env.windows.items():
        belief = window.belief.last
        mag = env._true_mag(agent)
        truth = (mag != 0) | (mag != 0).T
        got = belief.adjacency >= 0.5
        assert np.array_equal(got | got.T, truth), agent
        assert np.all(np.isin(belief.adjacency, (0.0, 1.0)))
        # No interventions yet, so no confounding may be claimed anywhere.
        assert not (belief.bidirected >= env.config.claim_bar).any()


def test_oracle_flag_refused_on_exact_backend():
    with pytest.raises(ValueError, match="oracle"):
        TwoAgentEnv(MAConfig(topology=TOPO, oracle_obs_structure=True), seed=0)


# =======================================================================================
# The vectorised tally must be the SAME function as the enumerator, not a second opinion.
#
# `score_window` stopped materialising Claim objects on 30 Aug 2026 -- it was 13% of the
# training wall clock, four calls per environment step. The readable definition stayed
# (`enumerate_claims`, which traces read); only the tally moved to arrays. Two definitions
# of one metric is how a metric drifts, so this asserts they agree EXACTLY, over
# randomised beliefs and MAGs, at every bar and both flag settings.
# =======================================================================================
import itertools

import numpy as np
import pytest

from cb.claims import ClaimScore, _tally, enumerate_claims
from ma.projection import BIDIRECTED as MAG_BIDIRECTED
from ma.projection import DIRECTED as MAG_DIRECTED


class _FreqBelief:
    def __init__(self, adjacency, directed, bidirected):
        self.adjacency, self.directed, self.bidirected = adjacency, directed, bidirected


def _reference(belief, mag, private, bar, require_all_types, confounding_claims):
    """The old body of `score_window`, kept here as the thing the fast path must match."""
    claims = enumerate_claims(belief, mag, private, bar,
                              require_all_types=require_all_types,
                              confounding_claims=confounding_claims)
    required = [c for c in claims if c.required]
    return ClaimScore(sum(c.outcome == "right" for c in claims),
                      sum(c.outcome == "wrong" for c in claims),
                      sum(c.outcome == "unsure" for c in claims),
                      sum(c.outcome == "right" for c in required), len(required),
                      sum(c.outcome == "wrong" for c in required))


def _random_case(rng, k):
    """A MAG and a belief whose frequencies land ON the bars as well as around them.

    Frequencies are drawn from {0, 1/4, 1/3, 1/2, 2/3, 3/4, 1} rather than uniformly:
    the outcome is a `>=` comparison against the bar, so the cases that separate two
    implementations are exact ties, and uniform draws never produce one.
    """
    levels = np.array([0.0, 0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75, 1.0])
    mag = np.zeros((k, k), dtype=np.int8)
    for u, v in itertools.combinations(range(k), 2):
        roll = rng.integers(0, 4)
        if roll == 1:
            mag[u, v] = MAG_DIRECTED
        elif roll == 2:
            mag[v, u] = MAG_DIRECTED
        elif roll == 3:
            mag[u, v] = mag[v, u] = MAG_BIDIRECTED
    pick = lambda: rng.choice(levels, size=(k, k))
    adjacency, directed, bidirected = pick(), pick(), pick()
    adjacency = np.triu(adjacency, 1); adjacency += adjacency.T
    bidirected = np.triu(bidirected, 1); bidirected += bidirected.T
    np.fill_diagonal(directed, 0.0)
    return mag, _FreqBelief(adjacency, directed, bidirected)


@pytest.mark.parametrize("bar", [0.5, 0.7, 1.0])
@pytest.mark.parametrize("require_all_types", [True, False])
@pytest.mark.parametrize("confounding_claims", [True, False])
def test_the_vectorised_tally_equals_the_enumerator(bar, require_all_types,
                                                    confounding_claims):
    rng = np.random.default_rng(20260830)
    for trial in range(40):
        k = int(rng.integers(2, 9))
        mag, belief = _random_case(rng, k)
        private = tuple(int(p) for p in rng.choice(k, size=int(rng.integers(0, k + 1)),
                                                   replace=False))
        fast = _tally(belief, mag, private, bar, require_all_types, confounding_claims)
        slow = _reference(belief, mag, private, bar, require_all_types, confounding_claims)
        assert fast == slow, (f"trial {trial}: k={k} private={private}\n"
                              f"  fast {fast}\n  slow {slow}")


def test_the_tally_is_defined_on_a_degenerate_one_node_window():
    """k=1 has no pairs, so no claims -- and `identified` must not read as False."""
    belief = _FreqBelief(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))
    score = _tally(belief, np.zeros((1, 1), dtype=np.int8), (), 1.0)
    assert score == ClaimScore(0, 0, 0, 0, 0, 0)
    assert score.identified
