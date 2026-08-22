"""PHASE 1 GATE -- the subset DP must reproduce the enumerated two-agent belief exactly.

Run against `tests/fixtures/ma_reference_posteriors.npz`, frozen in Phase 0 from the
enumerated path before any of this existed. The fixture stores the belief INPUTS as well as
the outputs, so the DP is fed byte-identical data and any discrepancy is the DP's.

The gate splits, deliberately and for a documented reason. POOLED, SUBSET and JOINT are
modular and are held to 1e-10. JOINT_CONF is not modular under the enumerated path's
definition -- it orients each confounding edge by an arbitrary topological tie-break -- so
the DP implements a reformulation that marginalises the orientation instead. It is checked
for internal consistency here and compared to the old rule by measurement, not identity.
See `ma/belief_dp.py`'s module docstring.

!! DO NOT MOVE THIS FILE TO legacy/tests/ !!

It imports from `legacy/`, which makes it look like the nineteen retired v1 test files moved
out on 2026-08-22. It is the opposite. Here v1 is the **independent reference oracle** for
CURRENT code: the value of the check is precisely that the reference shares no code with the
thing under test, so a shared bug cannot hide in both. If `legacy/ma_v1/` is ever deleted,
convert this to a frozen fixture FIRST -- never drop the check.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ma.belief_dp import JOINT, POOLED, SUBSET, WindowBeliefDP
from legacy.ma_v1.env import AgentView
from ma.topology import Topology

FIXTURE = pathlib.Path("tests/fixtures/ma_reference_posteriors.npz")
TOL = 1e-10
AGENTS = ("A", "B")
MODULAR = (POOLED, SUBSET, JOINT)


@pytest.fixture(scope="module")
def fixture():
    if not FIXTURE.exists():
        pytest.skip(f"{FIXTURE} missing -- run scripts/ma_freeze_reference.py")
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def topology():
    return Topology(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))


@pytest.fixture(scope="module")
def views(topology):
    return {name: AgentView(name, topology) for name in AGENTS}


@pytest.fixture(scope="module")
def beliefs(views, topology):
    out = {}
    for name, view in views.items():
        shared_positions = [view.pos[node] for node in view.shared]
        out[name] = WindowBeliefDP(view.k, shared_positions)
    return out


def enumerated_marginals(view, posterior: np.ndarray) -> np.ndarray:
    """Edge marginals implied by a full posterior over the window's DAGs."""
    return np.tensordot(posterior, view.dags.astype(float), axes=(0, 0))


def iter_entries(fixture, limit: int):
    """Yield (episode, round, agent) keys present in the fixture."""
    seen = 0
    for episode in range(1000):
        if f"e{episode}_truth" not in fixture:
            break
        for rnd in range(8):
            if f"e{episode}_r{rnd}_samples" not in fixture:
                break
            for name in AGENTS:
                yield episode, rnd, name
                seen += 1
                if seen >= limit:
                    return


def load(fixture, episode, rnd, name):
    tag = f"e{episode}_r{rnd}_{name}"
    nodes = fixture[f"{tag}_nodes"]
    samples = fixture[f"e{episode}_r{rnd}_samples"][:, nodes]
    return samples, fixture[f"{tag}_known"], fixture[f"{tag}_clean"]


@pytest.mark.parametrize("rule", MODULAR)
def test_dp_edge_marginals_match_the_enumerated_posterior(fixture, views, beliefs, rule):
    """The gate. Any failure here means the DP is not the same estimator."""
    worst = 0.0
    checked = 0
    for episode, rnd, name in iter_entries(fixture, limit=120):
        samples, known, clean = load(fixture, episode, rnd, name)
        reference = enumerated_marginals(
            views[name], fixture[f"e{episode}_r{rnd}_{name}_post_{rule}"])
        got = beliefs[name].edge_marginals(samples, known, clean, rule)
        worst = max(worst, float(np.abs(got - reference).max()))
        checked += 1
    assert checked > 0
    assert worst < TOL, f"{rule}: worst edge-marginal discrepancy {worst:.3e} over {checked}"


@pytest.mark.parametrize("rule", MODULAR)
def test_dp_dag_probability_matches_the_enumerated_posterior(fixture, views, beliefs, rule):
    """Marginals can agree while the joint disagrees -- ~10% of mass has sat on a wrong
    skeleton with every marginal looking correct. So the true DAG's probability is checked
    directly, not inferred from the marginals."""
    worst = 0.0
    for episode, rnd, name in iter_entries(fixture, limit=60):
        samples, known, clean = load(fixture, episode, rnd, name)
        view = views[name]
        index = int(fixture[f"e{episode}_true_index"][AGENTS.index(name)])
        reference = float(fixture[f"e{episode}_r{rnd}_{name}_post_{rule}"][index])
        got = float(np.exp(beliefs[name].log_prob_dag(
            samples, known, clean, rule, view.dags[index])))
        worst = max(worst, abs(got - reference))
    assert worst < TOL, f"{rule}: worst true-DAG probability discrepancy {worst:.3e}"


def test_confounding_is_confined_to_shared_pairs(views, beliefs):
    """The proved confinement result, enforced rather than assumed. A topology change that
    quietly let a private node into a confounding pair would otherwise pass silently."""
    for name, belief in beliefs.items():
        private_positions = {views[name].pos[node] for node in views[name].private}
        for u, v in belief.pairs:
            assert u not in private_positions and v not in private_positions


def test_cyclic_assignments_are_dropped(beliefs):
    """3 states per shared pair (absent, u->v, v->u) gives 3^3 = 27 at |X|=3, but the three
    shared pairs form a triangle and its 2 cyclic orientations admit no acyclic completion.
    Leaving them in makes the DP's inclusion-exclusion cancel to an exactly zero partition
    function -- which is how this was found."""
    for belief in beliefs.values():
        assert len(belief.pairs) == 3
        assert belief.n_assignments == 25, "27 assignments minus 2 cyclic orientations"


def test_joint_conf_marginals_are_a_valid_probability_field(fixture, beliefs):
    """Internal consistency, since identity with the old rule is not the target: every
    marginal in [0,1], no NaN, and no edge and its reverse both certain."""
    for episode, rnd, name in iter_entries(fixture, limit=20):
        samples, known, clean = load(fixture, episode, rnd, name)
        m = beliefs[name].joint_conf_marginals(samples, known, clean)
        assert np.isfinite(m).all()
        assert (m >= -TOL).all() and (m <= 1 + TOL).all()
        assert np.allclose(np.diag(m), 0.0, atol=TOL)
        assert (m + m.T <= 1 + 1e-9).all(), "an edge and its reverse cannot both be certain"


def test_the_dp_scales_past_enumeration():
    """Phase 1's third gate: if k=10 is not tractable the DP has not bought what it was
    chosen for, and the plan stops rather than proceeding on a false scaling claim."""
    import time

    k = 10
    belief = WindowBeliefDP(k, shared_positions=[0, 1])
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(300, k))
    known = np.zeros((300, k))
    clean = np.zeros(300, dtype=bool)

    started = time.time()
    marginals = belief.edge_marginals(samples, known, clean, POOLED)
    elapsed = time.time() - started

    assert marginals.shape == (k, k)
    assert np.isfinite(marginals).all()
    # 10 nodes is 4.2e18 DAGs; enumeration is not merely slow there, it is impossible.
    assert elapsed < 120, f"k=10 took {elapsed:.0f}s"
