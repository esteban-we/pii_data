"""Draw faceight: 1/8 of the corpus, unused episodes only, scene mix matching the full corpus.

Writes scene_category + faceight columns into episode_usage.csv and the id list to
faceight_episodes.txt. Deterministic for a given input CSV. See .knuth/docs/faceight.md.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_STORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * 3))
CSV = os.path.join(_STORE, "data", "episode_usage.csv")
LIST = os.path.join(_STORE, "data", "faceight", "episodes.txt")
SEED = 19760703
FRACTION = 8

# Regex over the 4th token of env_id (e.g. 0014_TSZAMK_05_Factory -> "factory"); first match wins.
RULES = [
    (r"fact|facat|facro|manufactur", "Factory"),
    (r"^home$", "Home"),
    (r"^workbench$", "Workbench"),
    (r"^supermarket$", "Supermarket"),
    (r"store|retail|shop$|^story$", "Store"),
    (r"craft|marquetry|mosaic|studio|handcraft", "CraftStudio"),
    (r"logistic|storage|fulfillment", "Logistics"),
    (r"health|hospital|pharmacy|clinic|beauty|nailsalon", "Healthcare"),
    (r"^offic", "Office"),
    (r"restaurant|coffee|food|bakery|chicken|hotel|hotal", "Restaurant"),
]
EXCLUDE_TOKENS = {"test", "internaltestonly", "demo"}


def category(tok: str) -> str:
    t = tok.lower()
    for pat, name in RULES:
        if re.search(pat, t):
            return name
    return "Other"


def main() -> None:
    d = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    tok = d.env_id.str.split("_").str[3].fillna("")
    d["scene_category"] = tok.map(category)

    n = round(len(d) / FRACTION)
    share = d.scene_category.value_counts(normalize=True).sort_index()
    raw = share * n
    quota = np.floor(raw).astype(int)
    for c in (raw - quota).sort_values(ascending=False).index[: n - quota.sum()]:
        quota[c] += 1

    pool = d[(d.role == "unused") & (d.session_id != "") & ~tok.str.lower().isin(EXCLUDE_TOKENS)]
    rng = np.random.default_rng(SEED)
    picked: set[str] = set()
    for c in sorted(quota.index):
        ids = sorted(pool.loc[pool.scene_category == c, "episode_id"])
        picked |= set(rng.choice(ids, quota[c], replace=False))
    d["faceight"] = d.episode_id.isin(picked).astype(int)

    d.to_csv(CSV, index=False)
    with open(LIST, "w") as fh:
        fh.write("\n".join(sorted(picked)) + "\n")

    got = d[d.faceight == 1].scene_category.value_counts()
    rep = pd.DataFrame({"corpus": d.scene_category.value_counts(), "corpus_%": (100 * share).round(2),
                        "quota": quota, "faceight_%": (100 * got / n).round(2)}).sort_values("corpus", ascending=False)
    print(rep.to_string())
    print(f"\nN={n} picked={len(picked)} seed={SEED} pool={len(pool)} -> {CSV}, {LIST}")


if __name__ == "__main__":
    main()
