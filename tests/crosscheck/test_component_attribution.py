"""The component-factored attribution belief against the enumerated one.

WHAT IS BEING ESTABLISHED. `cb/component_attribution.py` claims two things, and they are
tested separately because they can fail separately:

  1. THE SPACE FACTORS EXACTLY. The enumerated candidate set equals the product over
     connected components of the per-component enumerations. If this is false the whole
     design is unsound, so it is pinned directly rather than inferred from episode results.
  2. THE PRUNING IS CONSERVATIVE, NOT WRONG. Rule 1 can span components, and a clause across
     components is applied only when unit propagation makes it unit. The belief is therefore
     a superset of the enumerated one, so it may be LESS decided -- never differently decided,
     never wrong.

The episode assertions are one-sided for the same reason the existing crosscheck is: being
less decided is the documented price, and being differently decided is the defect.
"""
from __future__ import annotations

import itertools
import random

import pytest

from cb.attribution import (AttributedVersionSpaceBackend, LatentGroup, attributions_for,
                            score_groups)
from cb.component_attribution import ComponentAttributedBackend, connected_components
from cb.factored_attribution import FactoredAttributedBackend
from tests.crosscheck.test_factored_attribution import _drive, _env


def _canonical(groups):
    return tuple(sorted(groups, key=lambda g: (g.owner, sorted(g.children))))


@pytest.mark.parametrize("seed", range(4))
def test_the_attribution_space_is_exactly_the_product_over_components(seed):
    """The identity the design rests on: enumeration factors over connected components.

    Owner sets are assigned per pair, cliques never span components, and the coverage check
    is per pair -- so the joint enumeration should be reconstructible as a product with no
    candidate gained and none lost. Checked as SET EQUALITY, not as a count.
    """
    rng = random.Random(seed)
    checked = 0
    for _ in range(60):
        nodes = range(rng.randint(2, 7))
        every = list(itertools.combinations(nodes, 2))
        pairs = tuple(sorted(rng.sample(every, min(rng.randint(1, 5), len(every)))))
        owners = tuple(range(1, rng.randint(2, 4)))
        joint = set(attributions_for(pairs, owners))
        pieces = [attributions_for(component, owners)
                  for component in connected_components(pairs)]
        product = {_canonical([g for part in combination for g in part])
                   for combination in itertools.product(*pieces)}
        assert joint == product, (
            f"pairs={pairs} owners={owners}: the product over components differs from the "
            f"joint enumeration by {len(joint ^ product)} candidate(s) -- the factoring is "
            f"not exact and the backend built on it is unsound")
        checked += 1
    assert checked == 60


@pytest.mark.parametrize("episode", range(6))
def test_component_backend_never_contradicts_the_enumerated_backend(episode):
    """Never wrong, and never more decided than the belief it approximates."""
    env = _env()
    agents = list(env.topology.agents)
    fast = {a: ComponentAttributedBackend(env.windows[a].k, n_agents=len(agents), agent=a,
                                          evidence="oracle") for a in agents}
    _drive(env, fast, episode)

    env2 = _env()
    slow = {a: AttributedVersionSpaceBackend(env2.windows[a].k, n_agents=len(agents),
                                             agent=a) for a in agents}
    _drive(env2, slow, episode)

    for agent in agents:
        f = score_groups(fast[agent].last, fast[agent].true_groups, bar=1.0)
        s = score_groups(slow[agent].last, slow[agent].true_groups, bar=1.0)
        assert f["total"] == s["total"], "the two backends disagree on the TRUE groups"
        assert f["wrong"] == 0, (
            f"episode {episode}, agent {agent}: the component backend settled {f['wrong']} "
            f"attribution(s) WRONG -- the soundness guarantee has broken")
        settled_fast = {g for g, outcome, _ in f["detail"] if outcome == "right"}
        settled_slow = {g for g, outcome, _ in s["detail"] if outcome == "right"}
        assert settled_fast <= settled_slow, (
            f"episode {episode}, agent {agent}: the component backend settled "
            f"{settled_fast - settled_slow} that the enumerated one did not -- a superset "
            f"belief cannot be MORE decided than the belief it contains")


def test_component_backend_matches_the_enumerated_factored_one_where_both_run():
    """Against `FactoredAttributedBackend`, which is the same structure half.

    The two differ ONLY in how ownership is held, so any difference in the score is the price
    of the cross-component clauses this backend cannot represent -- reported here rather than
    left to be argued. The assertion stays one-sided; the number is printed so the loss is a
    measured quantity and not a claim.
    """
    agents_right = lost = 0
    for episode in range(8):
        env = _env()
        agents = list(env.topology.agents)
        fast = {a: ComponentAttributedBackend(env.windows[a].k, n_agents=len(agents),
                                              agent=a, evidence="oracle") for a in agents}
        _drive(env, fast, episode)
        env2 = _env()
        ref = {a: FactoredAttributedBackend(env2.windows[a].k, n_agents=len(agents), agent=a,
                                            evidence="oracle") for a in agents}
        _drive(env2, ref, episode)
        for agent in agents:
            f = score_groups(fast[agent].last, fast[agent].true_groups, bar=1.0)
            r = score_groups(ref[agent].last, ref[agent].true_groups, bar=1.0)
            assert f["wrong"] == 0
            got = {g for g, outcome, _ in f["detail"] if outcome == "right"}
            had = {g for g, outcome, _ in r["detail"] if outcome == "right"}
            agents_right += len(got)
            lost += len(had - got)
            assert not (got - had), (
                f"episode {episode}, agent {agent}: component backend settled {got - had} "
                f"that the enumerated-ownership backend did not")
    print(f"\ncomponent-factored: {agents_right} settled right, {lost} decisions given up "
          f"to cross-component clauses")


def test_no_partner_evidence_means_unsure_never_wrong():
    """The soundness floor. A confident misattribution is worse than no attribution."""
    env = _env()
    agents = list(env.topology.agents)
    backends = {a: ComponentAttributedBackend(env.windows[a].k, n_agents=len(agents),
                                              agent=a, evidence="oracle") for a in agents}
    total = wrong = 0
    for episode in range(8):
        result = env.reset(seed=100 + episode)
        for agent, backend in backends.items():
            backend.reset(env._true_mag(agent), adjacency=env.true_adjacency,
                          topology=env.topology)
        turns = {a: 0 for a in agents}
        while not result.done:
            active = env.active_agent()
            result = env.step({a: env.windows[a].action_index(
                                   env.windows[a].nodes[turns[a] % env.windows[a].k], "vary")
                               for a in agents})
            for agent, backend in backends.items():
                backend.edge_marginals(env.samples[:, env.windows[agent].nodes],
                                       env.known[agent])
            if active is not None:
                turns[active] += 1
        for agent, backend in backends.items():
            score = score_groups(backend.last, backend.true_groups, bar=1.0)
            total += score["total"]
            wrong += score["wrong"]
    assert total > 0, "no true groups in any episode -- the test proves nothing"
    assert wrong == 0, (f"{wrong} of {total} true groups settled WRONG with no partner "
                        f"evidence at all")
