#!/usr/bin/env python3
"""
primary_models.py - condition effects on the session-level brain-state measures.

Input : <base>/pathA_results/01_session_metrics.csv
Output: <out>/tables/
    table3_mixed_models.csv          contrasts vs HFS (Dunnett) and among non-HFS conditions (Holm),
                                     condition-by-visit likelihood-ratio test
    table3_sensitivity_v2.csv        the same contrasts with the v2-inverted measures (*_alt columns)
    table4_per_visit_hfs_vs_vfs.csv  exploratory paired t-tests per visit, BH-FDR across visits
    condition_means.csv              descriptive condition means

Usage: python scripts/primary_models.py --base <data folder> --out <output folder>
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2, ttest_rel

from npj1_stats import (CONDITIONS, VISITS, CF, VF, fit, dunnett, pairwise_holm, condition_model, bh)


def load(base):
    d = pd.read_csv(Path(base) / "pathA_results" / "01_session_metrics.csv")
    d = d.rename(columns={"subject": "patient", "timepoint": "visit"})
    for s in ["S1", "S2", "S3", "S4"]:
        if s in d:
            d[f"{s}_pct"] = d[s] * 100
        if f"{s}_alt" in d:
            d[f"{s}_pct_alt"] = d[f"{s}_alt"] * 100
    return d


def outcomes(d):
    """Primary, secondary and follow-up outcomes present in the file."""
    wanted = ["dist_HC", "S4_residence_TR", "S4_pct", "static_fc"]
    extra = [c for c in d.columns if c.startswith("dist_") and not c.endswith("_alt") and c != "dist_HC"]
    return [c for c in wanted + extra if c in d.columns]


def interaction_lrt(d, outcome):
    """Condition-by-visit interaction: likelihood-ratio test of ML fits with and without it."""
    x = d[d.condition.isin(CONDITIONS)].dropna(subset=[outcome]).copy()
    x["visit"] = pd.Categorical(x["visit"], VISITS)
    full = fit(f"{outcome} ~ {CF} * {VF}", x, reml=False)
    red = fit(f"{outcome} ~ {CF} + {VF}", x, reml=False)
    stat = 2 * (full.llf - red.llf)
    df = len(full.fe_params) - len(red.fe_params)
    return stat, df, chi2.sf(stat, df)


ALL_PAIRS = (("VFS", "HFS"), ("LFS", "HFS"), ("OFF", "HFS"), ("VFS", "LFS"), ("VFS", "OFF"), ("LFS", "OFF"))


def contrasts(d, outcome, label=None):
    res, x = condition_model(d, outcome)
    if outcome == "static_fc":       # exploratory benchmark: all six pairs in one Holm family
        t = pairwise_holm(res, pairs=ALL_PAIRS).assign(method="Holm (6 pairs)")
    else:
        a = dunnett(res).assign(method="Dunnett")
        b = pairwise_holm(res).assign(method="Holm")
        t = pd.concat([a, b], ignore_index=True)
    t.insert(0, "outcome", label or outcome)
    t["n_obs"], t["n_patients"] = len(x), x.patient.nunique()
    t["random_intercept_var"] = float(res.cov_re.iloc[0, 0])
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="folder that contains pathA_results/")
    ap.add_argument("--out", required=True, help="output folder")
    a = ap.parse_args()
    out = Path(a.out) / "tables"; out.mkdir(parents=True, exist_ok=True)
    d = load(a.base)

    rows, lrt = [], []
    for o in outcomes(d):
        rows.append(contrasts(d, o))
        s, df, p = interaction_lrt(d, o)
        lrt.append(dict(outcome=o, contrast="condition x visit", chi2=s, df=df, p_raw=p))
    t3 = pd.concat(rows + [pd.DataFrame(lrt)], ignore_index=True)
    t3.to_csv(out / "table3_mixed_models.csv", index=False)

    alt = [o for o in outcomes(d) if f"{o}_alt" in d.columns]
    if alt:
        sens = [contrasts(d.drop(columns=[o]).rename(columns={f"{o}_alt": o}), o, f"{o} (v2 inverted)") for o in alt]
        pd.concat(sens, ignore_index=True).to_csv(out / "table3_sensitivity_v2.csv", index=False)

    # Table 4: exploratory per-visit HFS vs VFS (paired), BH-FDR across the four visits
    t4 = []
    for col in [c for c in ["dist_HC", "dist_HC_alt"] if c in d]:
        recs = []
        for v in VISITS:
            w = d[(d.visit == v) & d.condition.isin(["HFS", "VFS"])].pivot_table(
                index="patient", columns="condition", values=col).dropna()
            diff = w["VFS"] - w["HFS"]
            t, p = ttest_rel(w["VFS"], w["HFS"])
            recs.append(dict(measure=col, visit=v, n=len(w), mean_HFS=w["HFS"].mean(), mean_VFS=w["VFS"].mean(),
                             diff_VFS_minus_HFS=diff.mean(), cohens_dz=abs(diff.mean() / diff.std(ddof=1)),
                             t=t, p_raw=p))
        r = pd.DataFrame(recs); r["p_FDR"] = bh(r.p_raw); t4.append(r)
    if t4:
        pd.concat(t4).to_csv(out / "table4_per_visit_hfs_vs_vfs.csv", index=False)

    means = d[d.condition.isin(CONDITIONS + ["PREOP"])].groupby("condition")[outcomes(d)].mean()
    means.to_csv(out / "condition_means.csv")

    pd.set_option("display.width", 160)
    show = t3[t3.method.notna()][["outcome", "contrast", "beta", "ci_lo", "ci_hi", "p_raw", "p_adj", "method"]]
    print("\nTable 3 (contrasts):\n", show.round(4).to_string(index=False))
    print("\nCondition x visit:\n", pd.DataFrame(lrt).round(4).to_string(index=False))
    print(f"\nTables written to {out}")


if __name__ == "__main__":
    main()
