#!/usr/bin/env python3
"""
make_demo_data.py - writes a SYNTHETIC data set with the same file layout as the real one.

The values are random numbers with a built-in HFS effect. They are not patient data and
have no scientific meaning; they only let users test that every script runs end to end.

Usage: python scripts/make_demo_data.py --out demo_data
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

VISITS = ["1m", "3m", "6m", "12m"]
CONDS = ["OFF", "LFS", "HFS", "VFS"]
CODES = {"OFF": "OFF", "LFS": "CL", "HFS": "CH", "VFS": "CV"}


def occupancy(rng, mean, conc=300):
    return rng.dirichlet(np.asarray(mean) * conc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_data")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); res = out / "pathA_results"; clin = out / "clinical_processed"
    res.mkdir(parents=True, exist_ok=True); clin.mkdir(parents=True, exist_ok=True)

    hc_mean = np.array([0.26, 0.23, 0.27, 0.24])
    hc = pd.DataFrame([occupancy(rng, hc_mean, 80) for _ in range(15)], columns=["S1", "S2", "S3", "S4"])
    hc.insert(0, "subject", [f"HC{i + 1:02d}" for i in range(15)])
    hc.to_csv(res / "05_hc_occupancy.csv", index=False)
    ref = hc[["S1", "S2", "S3", "S4"]].mean().values

    pats = [f"sub-DEMO{i + 1:02d}" for i in range(14)]
    last_visit = {pats[0]: "1m", pats[1]: "3m"}            # incomplete follow-up, as in a real cohort
    base_occ = np.array([0.165, 0.07, 0.30, 0.465])
    rows = []
    for p in pats:
        u = rng.normal(0, 0.01, 4)                          # patient-specific offset
        def session(cond, visit, shift):
            m = np.clip(base_occ + u + shift, 0.01, None); m /= m.sum()
            occ = occupancy(rng, m)
            alt = occupancy(rng, np.clip(occ + rng.normal(0, 0.01, 4), 0.01, None) / 1.0)
            res_tr = 2.10 + (0.08 if cond == "HFS" else 0) + rng.normal(0, 0.08)
            return dict(subject=p, timepoint=visit, condition=cond,
                        **{f"S{k + 1}": occ[k] for k in range(4)}, **{f"S{k + 1}_alt": alt[k] for k in range(4)},
                        dist_HC=np.linalg.norm(occ - ref), dist_HC_alt=np.linalg.norm(alt - ref),
                        S4_residence_TR=res_tr, S4_residence_TR_alt=res_tr + rng.normal(0, 0.05),
                        static_fc=0.235 + rng.normal(0, 0.012))
        rows.append(session("PREOP", "preop", 0))
        for v in VISITS:
            if p == pats[2] and v == "1m":
                continue
            if p in last_visit and VISITS.index(v) > VISITS.index(last_visit[p]):
                continue
            for c in CONDS:
                rows.append(session(c, v, np.array([-0.005, -0.005, -0.01, 0.02]) if c == "HFS" else 0))
    sm = pd.DataFrame(rows)
    sm.to_csv(res / "01_session_metrics.csv", index=False)

    rois = [f"ROI_{i + 1:02d}" for i in range(93)]
    cen = pd.DataFrame(rng.normal(0, 0.03, (4, 93)), columns=rois)
    cen.insert(0, "state", ["S1", "S2", "S3", "S4"])
    cen.to_csv(res / "04_centroids.csv", index=False)

    tr = []
    for c in CONDS:
        P = rng.dirichlet(np.ones(4) * 5, 4) * 0.5 + np.eye(4) * 0.5
        if c == "HFS":
            P[3] = P[3] * 0.97 + np.eye(4)[3] * 0.03
        for i in range(4):
            for j in range(4):
                tr.append(dict(condition=c, from_state=f"S{i + 1}", to_state=f"S{j + 1}", probability=P[i, j]))
    pd.DataFrame(tr).to_csv(res / "13_fig5B_transition_matrices.csv", index=False)

    scales = {"III": (37, 14, 20), "bradykinesia": (16, 6, 9), "rigidity": (9, 3, 5),
              "PIGD": (5, 1.5, 2), "tremor": (2, 0.6, 0.8)}
    sess = sm[["subject", "timepoint", "condition"]]
    for dom, (off, hfs, other) in scales.items():
        recs = []
        for r in sess.itertuples():
            mean = {"PREOP": off * 0.9, "OFF": off, "HFS": hfs}.get(r.condition, other)
            true = max(0, rng.normal(mean, mean * 0.3))
            r1, r2 = [max(0, round(true + rng.normal(0, mean * 0.08), 1)) for _ in range(2)]
            if dom == "rigidity":
                r2 = r1
            cond = "preop" if r.condition == "PREOP" else f"{r.timepoint}{CODES[r.condition]}"
            recs.append(dict(participant_id=r.subject, type=dom, condition=cond,
                             rater1=r1, rater2=r2, aver=(r1 + r2) / 2))
        pd.DataFrame(recs).to_csv(clin / f"UPDRS_{dom}_processed.tsv", sep="\t", index=False)
    print(f"Synthetic demo data written to {out}/ ({len(sm)} synthetic sessions, {len(pats)} synthetic patients)")


if __name__ == "__main__":
    main()
