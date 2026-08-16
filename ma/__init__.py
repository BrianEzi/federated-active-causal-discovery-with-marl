"""Two-agent federated active causal discovery.

Separate from `sa/` for the same reason `sa/` was separate from `src/`: the single-agent
results are established and must not shift because something here was changed. This package
imports from `sa/` freely -- the score, the subset DP, the sampler and the oracle are all
agent-agnostic -- but nothing in `sa/` imports from here.
"""
