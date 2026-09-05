"""Scan thesis prose for phrases the CLAIMS file forbids.

The MUST NOT lines in thesis_results/CLAIMS.md are hand-maintained boundaries; this script
is their enforcement arm for the final sweep (docs/ENDGAME_2026_09_07.md, Monday). It scans
NON-COMMENT prose only: the commented analysis bullets legitimately discuss forbidden
phrasings while telling Brian not to use them.

Each rule is (pattern, why). Patterns are case-insensitive regex over comment-stripped text
per file. Exit 1 on any hit so it can gate the Monday build.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = ["thesis/0 Abstract.tex", "thesis/1 Introduction.tex",
         "thesis/2 Background and Related Work.tex", "thesis/3 Methodology.tex",
         "thesis/4 Results and Analysis.tex", "thesis/5 Discussion.tex",
         "thesis/6 Conclusion.tex", "thesis/Appendix.tex",
         "thesis/Results Tables.tex", "thesis/Negative Results.tex"]

# Each rule: (pattern, why, exempt_near) -- a hit within 300 chars of exempt_near is
# allowed, because the withdrawal record legitimately NAMES the claims it withdraws, and
# generic uses of 'centralised' (the literature taxonomy, 'no centralised critic', 'a
# genuinely centralised controller') are not the arm-E naming the 3 Sep rename retired.
RULES = [
    (r"neglect", "C3: 'neglects' is retracted wording for the unrewarded class",
     r"withdraw|retract|Withdrawn|refuted"),
    (r"15 of 18|15/18", "C7: it is 14 cells changing winner plus one exact tie, never 15",
     None),
    (r"agent-count reversal(?!.*(not|artefact|gone|withdraw))",
     "C2: no reversal exists at the converged budget", r"withdraw|retract"),
    (r"exactly unchanged", "C3a: Poisson SE ~3.3 on 11 events; say 'no detectable change'",
     None),
    (r"[Ll]earned \(centralised\)|[Cc]entralised (arm|run\b)|federated (\$-\$|minus|-) ?centralised|against centralised",
     "renamed 3 Sep: arm E is 'pooled' -- these are arm-E usages of 'centralised'", None),
    (r"more data hurts(?!.*fixed)", "C8: only with 'under fixed-alpha tests at evaluation'",
     None),
    (r"saturat\w+ (transfer|curve)", "C6: the plateau is not established as real", None),
    (r"3\.5\\times|3\.5x", "stale contribution ratio from the pre-rewrite Introduction",
     None),
    (r"ratio of means at \$?k_v ?= ?30|k30 ratio", "C1: one seed carries it", None),
]


def strip_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("%"))


def main() -> int:
    bad = 0
    for rel in FILES:
        path = ROOT / rel
        if not path.exists():
            continue
        prose = strip_comments(path.read_text())
        prose = re.sub(r"[Dd]ecentralis\w+", "", prose)
        for pattern, why, exempt in RULES:
            for m in re.finditer(pattern, prose, re.I):
                lo, hi = max(0, m.start() - 800), m.end() + 800
                if exempt and re.search(exempt, prose[lo:hi], re.I):
                    continue
                start = max(0, m.start() - 60)
                print(f"!! {rel}: /{pattern}/ -- {why}\n     ...{prose[start:m.end()+60]}...")
                bad += 1
    print(f"{'FAIL' if bad else 'CLEAN'}: {bad} forbidden-phrase hits")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
