"""Mechanical gate for the prose rules in thesis/WRITING_GUIDELINES.md.

Companion to scripts/check_mustnots.py, which gates FACTUAL claims. This one gates STYLE.
Both were written because the same defects kept recurring and neither the author nor the
reviewer caught them reliably by reading.

Every rule below traces to a numbered item in thesis/WRITING_CRITIQUE.md or a dated section
of WRITING_GUIDELINES.md. Scans non-comment prose only: commented scaffolding legitimately
discusses phrasings the prose may not use.

Density rules are per 1000 words, so a chapter is not penalised for being long. Presence
rules fire on any occurrence.

    .venv/bin/python scripts/check_style.py            # all chapters
    .venv/bin/python scripts/check_style.py --file "thesis/3 Methodology.tex"
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

CHAPTERS = ["thesis/0 Abstract.tex", "thesis/1 Introduction.tex",
            "thesis/2 Background and Related Work.tex", "thesis/3 Methodology.tex",
            "thesis/4 Results and Analysis.tex", "thesis/5 Discussion.tex",
            "thesis/6 Conclusion.tex"]

# (pattern, why, critique item). Presence rules -- any hit is reported.
PHRASES = [
    (r"it is worth (stating|noting)", "justifying the act of writing", "critique 2"),
    (r"\bmust be stated\b", "justifying the act of writing", "critique 2"),
    (r"it should be noted", "justifying the act of writing", "critique 2"),
    (r"which is (what|why|exactly)", "sentence restating its own significance", "critique 7"),
    (r"\bthat is deliberate\b", "concessive throat-clearing", "critique 9"),
    (r"this is not (a hedge|cosmetic|hedging|a neutral)", "concessive throat-clearing", "critique 9"),
    (r"\b(Two|Three|Four) (reasons|things|points|consequences|separate)\b",
     "announced enumeration", "critique 1"),
    (r"\bstated rather than buried\b", "concessive throat-clearing", "critique 9"),
    (r"\b(genuinely|essentially|actually|considerably|substantially)\b",
     "hedged intensifier doing no work", "critique 13"),
]

# Density rules: (name, counter, per-1000-word ceiling, why)
def _count(pat):
    return lambda t: len(re.findall(pat, t))

DENSITY = [
    ("em-dash aside", _count(r"---"), 3.0, "critique 4: most become their own sentence"),
    (r"\emph{}", _count(r"\\emph\{"), 2.0, "critique 6: let word order carry the stress"),
    (r"\textbf{}", _count(r"\\textbf\{"), 2.0, "critique 5: bold marks a defined term only"),
    ("'rather than'", _count(r"rather than"), 2.0,
     "critique 3: state the positive claim"),
]

# At least four letters before the suffix, so "size"/"sized"/"sizes" (correct BrEng, and
# half of LaTeX's font commands) do not match while "organize"/"characterize" do. LaTeX
# control sequences are stripped before this runs -- \centering was matching "center".
AMERICAN = re.compile(r"\b\w{4,}(?:ize|ized|izes|izing|ization|yze|yzed|yzes)\b"
                      r"|\bbehavior\b|\bcolor\b|\bfavor\b|\bcenter\b|\bneighbor\b", re.I)
AMERICAN_OK = re.compile(r"Optimization|Generalized|Recognize", re.I)


def strip_comments(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("%"))


def check(rel: str, methodology: bool) -> int:
    path = ROOT / rel
    if not path.exists():
        return 0
    prose = strip_comments(path.read_text())
    words = max(len(prose.split()), 1)
    # LaTeX control sequences are not prose: \centering must not read as "center", and
    # \footnotesize must not read as an -ize spelling.
    bare = re.sub(r"\\[A-Za-z]+", " ", prose)
    hits = 0
    for pat, why, item in PHRASES:
        for m in re.finditer(pat, prose, re.I):
            lo = max(0, m.start() - 55)
            print(f"  ! {why} [{item}]\n      ...{prose[lo:m.end()+55]}...")
            hits += 1
    for name, fn, ceiling, why in DENSITY:
        n = fn(prose)
        per_k = 1000.0 * n / words
        if per_k > ceiling:
            print(f"  ! {name}: {n} in {words} words = {per_k:.1f}/1000, ceiling {ceiling} "
                  f"({why})")
            hits += 1
    for m in AMERICAN.finditer(bare):
        seg = bare[max(0, m.start() - 30):m.end() + 30]
        if AMERICAN_OK.search(seg):
            continue
        print(f"  ! American spelling '{m.group(0)}' (guidelines: British English only)")
        hits += 1
    if methodology:
        # "Numbers in Methodology", added 6 Sep: design constants yes, measured outcomes no.
        # A bare large number cannot be classified mechanically -- 12,000 episodes is the
        # design, 157,680 pair instances is a result, and both are five digits. The count is
        # ADVISORY: printed for a human to judge, and it does not fail the gate. The phrase
        # patterns below are the reliable signal and those do fail it.
        advisory = []
        for m in re.finditer(r"\$?\d{1,3}\{,\}\d{3}\$?|\b\d{5,}\b", prose):
            val = m.group(0)
            if re.fullmatch(r"\$?\b(19|20)\d{2}\b\$?", val):
                continue
            advisory.append(val)
        if advisory:
            uniq = sorted(set(advisory))
            print(f"  . advisory, judge by hand: large numbers present {uniq} "
                  f"(design constant, or a result that belongs in Ch4/Ch5?)")
        for pat in (r"measured over", r"across [\d,{}$\\]+ (instances|episodes|pairs)",
                    r"\bwe measured\b"):
            for m in re.finditer(pat, prose, re.I):
                print(f"  ! '{m.group(0)}' in Methodology -- results belong in Ch4/Ch5")
                hits += 1
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=None)
    args = ap.parse_args(argv)
    files = [args.file] if args.file else CHAPTERS
    total = 0
    for rel in files:
        if not (ROOT / rel).exists():
            continue
        print(f"\n== {rel}")
        n = check(rel, methodology="Methodology" in rel)
        print("   clean" if n == 0 else f"   {n} issue(s)")
        total += n
    print(f"\n{'STYLE CLEAN' if total == 0 else f'STYLE: {total} issue(s)'}")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
