"""Frozen reference implementations. NOT production code.

Everything here exists to CHECK the live engine in `ma/` and `cb/`, never to be
extended. The exact Bayesian belief (`belief_dp`), the subset DP and BGe score it
rests on, the pre-DP regime scorer, and the v1 enumerated oracle that the Phase 1
gate is held to. The directory name is the point: it cannot be imported by accident
and read as the current path.
"""
