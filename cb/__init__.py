"""Constraint-based causal discovery for the multi-agent setting.

Independence tests instead of graph scoring. Replaces the exact Bayesian belief in
`crosscheck/belief_dp.py` for three measured reasons -- see docs/CB_IMPLEMENTATION_PLAN.md
section 2: it is tens of times faster at realistic window sizes, the independence test is a
plug-in so nonlinearity is a swap rather than a rewrite, and it has no clean/dirty score
mixture, which is what makes rung 1 (three agents) runnable at all.

    citest      the only contact with data. Swap this for a kernel test to go nonparametric.
    skeleton    adjacency search. Dominates runtime; records separating sets.
    orient      colliders + Meek + interventional orientation. Sound, not complete.
    bootstrap   resample -> distribution over graphs. What the policy consumes.
    backend     adapter to the belief-backend protocol in ma/env.py.
"""
