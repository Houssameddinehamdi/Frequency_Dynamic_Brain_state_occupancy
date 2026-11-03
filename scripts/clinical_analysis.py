#!/usr/bin/env python3
"""
clinical_analysis.py - exploratory clinical analyses.

  D.  Inter-rater reliability: ICC(2,1) between the two UPDRS-III raters, per domain.
  A.  Clinical response by domain: score ~ condition + visit + (1 | patient);
      contrasts vs HFS (Dunnett) and among non-HFS conditions (Holm).
      Domains: UPDRS-III total, bradykinesia, rigidity, PIGD, tremor.
  B.  Stimulation-induced coupling (each LFS/HFS/VFS session minus the same-visit DBS-OFF session):
      dSymptom ~ dBrain [+ condition] + visit + (1 | patient).
      Brain measures: d_HC, State 4 occupancy, State 4 residence time.
      Domains: rigidity, tremor, PIGD, bradykinesia. 12 tests per model, BH-FDR.
      The unadjusted model gives the between-condition association; the condition-adjusted
      model gives the within-condition (patient-level) association.
  B2. Does coupling differ by frequency? dSymptom ~ dBrain * condition + visit + (1 | patient);
      2-df Wald test of the interaction, BH-FDR.
  C.  Change from the presurgical assessment in HFS sessions: dSymptom ~ dBrain + visit + (1 | patient).
  Sensitivity: B, B2 and C repeated with the v2-inverted brain measures (*_alt columns).
Lower UPDRS scores indicate milder symptoms. Tremor scores are zero in about half of the sessions.

Inputs : <base>/pathA_results/01_session_metrics.csv
         <base>/clinical_processed/UPDRS_<domain>_processed.tsv
Outputs: <out>/clinical_analysis/*.csv

Usage: python scripts/clinical_analysis.py --base <data folder> --out <output folder>
"""
import argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import multivariate_normal, norm

warnings.filterwarnings("ignore")
_ap = argparse.ArgumentParser()
_ap.add_argument("--base", required=True, help="folder that contains pathA_results/ and clinical_processed/")
_ap.add_argument("--out", required=True, help="output folder")
_args = _ap.parse_args()
BASE = Path(_args.base)
CLIN = BASE / "clinical_processed"
OUT = Path(_args.out) / "clinical_analysis"
OUT.mkdir(parents=True, exist_ok=True)

DOMAINS_A = {"UPDRS_III": "UPDRS_III", "bradykinesia": "UPDRS_bradykinesia",
             "rigidity": "UPDRS_rigidity", "PIGD": "UPDRS_PIGD", "tremor": "UPDRS_tremor"}
DOMAINS_BC = ["rigidity", "tremor", "PIGD", "bradykinesia"]
BRAIN = {"dist_HC": "dist_HC", "S4_pp": "S4", "S4_residence_TR": "S4_residence_TR"}
CODE = {"OFF": "OFF", "CL": "LFS", "CH": "HFS", "CV": "VFS"}
VISITS = ["1m", "3m", "6m", "12m"]


# ---------------------------------------------------------------- helpers
def fit(formula, data, group="patient"):
    m = smf.mixedlm(formula, data, groups=data[group])
    try:
        r = m.fit(reml=True, method="powell")
    except Exception:
        r = m.fit(reml=True, method="lbfgs")
    return r


def dunnett(res, names):
    """Asymptotic Dunnett-adjusted p and simultaneous 95% CI for coefficients `names`."""
    b = res.fe_params[names].values
    V = res.cov_params().loc[names, names].values
    se = np.sqrt(np.diag(V))
    z = b / se
    R = V / np.outer(se, se)
    k = len(b)

    def box(c):
        return multivariate_normal.cdf(np.full(k, c), mean=np.zeros(k), cov=R, allow_singular=True,
                                       maxpts=2_000_000, abseps=1e-7, releps=1e-7, lower_limit=np.full(k, -c))
    p_adj = [1 - box(abs(zi)) for zi in z]
    # critical value for simultaneous 95% CI
    lo, hi = 1.5, 4.0
    for _ in range(60):
        c = (lo + hi) / 2
        lo, hi = (c, hi) if box(c) < 0.95 else (lo, c)
    c = (lo + hi) / 2
    return b, se, z, np.clip(p_adj, 0, 1), b - c * se, b + c * se, c


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.empty(n); run = 0
    for i, k in enumerate(o):
        run = max(run, (n - i) * p[k]); adj[k] = min(1, run)
    return adj


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.empty(n); run = 1
    for i in range(n - 1, -1, -1):
        k = o[i]; run = min(run, p[k] * n / (i + 1)); adj[k] = run
    return adj


def icc21(x, y):
    X = np.column_stack([x, y]).astype(float); n, k = X.shape
    gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def cname(level):
    return f"C(condition, Treatment('HFS'))[T.{level}]"


# ---------------------------------------------------------------- load
img = pd.read_csv(BASE / "pathA_results" / "01_session_metrics.csv")
img = img.rename(columns={"subject": "patient", "timepoint": "visit"})
img["condition"] = img["condition"].replace({"PREOP": "PREOP"})
img["S4_pp"] = img["S4"] * 100
img["S4_pp_alt"] = img["S4_alt"] * 100

clin = []
for dom, stem in DOMAINS_A.items():
    d = pd.read_csv(CLIN / f"{stem}_processed.tsv", sep="\t")
    for _, r in d.iterrows():
        c = str(r["condition"])
        if c == "preop":
            v, cond = "preop", "PREOP"
        else:
            v, code = c[:c.index("m") + 1], c[c.index("m") + 1:]
            if code not in CODE:
                continue
            cond = CODE[code]
        clin.append(dict(patient=r["participant_id"], visit=v, condition=cond, domain=dom,
                         score=r["aver"], rater1=r["rater1"], rater2=r["rater2"]))
clin = pd.DataFrame(clin)
wide = clin.pivot_table(index=["patient", "visit", "condition"], columns="domain",
                        values="score").reset_index()
df = img.merge(wide, on=["patient", "visit", "condition"], how="left")
print(f"Merged sessions: {len(df)}; with UPDRS-III: {df['UPDRS_III'].notna().sum()}")

post = df[df.condition != "PREOP"].copy()
post["visit"] = pd.Categorical(post["visit"], VISITS)
VF = "C(visit, Treatment('1m'))"
CF = "C(condition, Treatment('HFS'))"

# ---------------------------------------------------------------- 0. check
r = fit(f"dist_HC ~ {CF} + {VF}", post)
b, se, z, p, lo, hi, c = dunnett(r, [cname(l) for l in ["VFS", "LFS", "OFF"]])
print("\n[0] Check: d_HC contrasts vs HFS (should match the primary analysis):")
for l, bi, pi in zip(["VFS", "LFS", "OFF"], b, p):
    print(f"    {l}-HFS  beta={bi:+.4f}  Dunnett p={pi:.4f}")
print(f"    critical |z| = {c:.3f}")

# ---------------------------------------------------------------- D. ICC
rows = []
for dom in DOMAINS_A:
    s = clin[(clin.domain == dom)].dropna(subset=["rater1", "rater2"])
    rows.append(dict(domain=dom, n_sessions=len(s), ICC21=icc21(s.rater1, s.rater2),
                     mean_abs_diff=(s.rater1 - s.rater2).abs().mean(),
                     pct_zero=(s.score == 0).mean() * 100))
resD = pd.DataFrame(rows); resD.to_csv(OUT / "D_interrater_icc.csv", index=False)
print("\n[D] Inter-rater reliability\n", resD.round(3).to_string(index=False))

# ---------------------------------------------------------------- A. clinical phenotype
rowsA, meansA = [], []
for dom in DOMAINS_A:
    d = post.dropna(subset=[dom])
    r = fit(f"{dom} ~ {CF} + {VF}", d)
    names = [cname(l) for l in ["VFS", "LFS", "OFF"]]
    b, se, z, p, lo, hi, _ = dunnett(r, names)
    for l, *vals in zip(["VFS", "LFS", "OFF"], b, se, p, lo, hi):
        rowsA.append(dict(domain=dom, contrast=f"{l} - HFS", beta=vals[0], se=vals[1],
                          p_adj=vals[2], ci_lo=vals[3], ci_hi=vals[4], method="Dunnett",
                          n_obs=len(d), n_patients=d.patient.nunique()))
    # non-HFS pairs, Holm
    fe, V = r.fe_params, r.cov_params()
    pr, tmp = [], []
    for a, bb in [("VFS", "LFS"), ("VFS", "OFF"), ("LFS", "OFF")]:
        L = pd.Series(0.0, index=fe.index); L[cname(a)] = 1; L[cname(bb)] = -1
        est = float(L @ fe); s = float(np.sqrt(L @ V.loc[fe.index, fe.index] @ L))
        pr.append(2 * norm.sf(abs(est / s))); tmp.append((a, bb, est, s))
    for (a, bb, est, s), pa in zip(tmp, holm(pr)):
        rowsA.append(dict(domain=dom, contrast=f"{a} - {bb}", beta=est, se=s, p_adj=pa,
                          method="Holm", n_obs=len(d), n_patients=d.patient.nunique()))
    m = d.groupby("condition")[dom].mean()
    pre = df[df.condition == "PREOP"][dom].mean()
    meansA.append(dict(domain=dom, preop=pre, **m.to_dict()))
resA = pd.DataFrame(rowsA); resA.to_csv(OUT / "A_condition_effects_by_domain.csv", index=False)
resMA = pd.DataFrame(meansA); resMA.to_csv(OUT / "A_condition_means_by_domain.csv", index=False)
print("\n[A] Condition means (lower = better)\n", resMA.round(2).to_string(index=False))
print("\n[A] Contrasts\n", resA[["domain", "contrast", "beta", "p_adj", "method"]]
      .round(4).to_string(index=False))


# ---------------------------------------------------------------- B. within-visit coupling
def run_B(suffix=""):
    rows = []
    base = post.set_index(["patient", "visit", "condition"])
    off = post[post.condition == "OFF"].set_index(["patient", "visit"])
    stim = post[post.condition.isin(["LFS", "HFS", "VFS"])].copy()
    for dom in DOMAINS_BC:
        for key, col in BRAIN.items():
            bc = ("S4_pp" if key == "S4_pp" else col) + suffix
            recs = []
            for _, s in stim.iterrows():
                k = (s.patient, s.visit)
                if k not in off.index:
                    continue
                o = off.loc[k]
                if pd.isna(s[dom]) or pd.isna(o[dom]):
                    continue
                recs.append(dict(patient=s.patient, visit=s.visit, condition=s.condition,
                                 dS=s[dom] - o[dom], dN=s[bc] - o[bc]))
            d = pd.DataFrame(recs)
            d["visit"] = pd.Categorical(d["visit"], VISITS)
            for adj in ["adjusted", "unadjusted"]:
                f = (f"dS ~ dN + {CF} + {VF}" if adj == "adjusted" else f"dS ~ dN + {VF}")
                r = fit(f, d)
                beta, se, pv = r.fe_params["dN"], r.bse["dN"], r.pvalues["dN"]
                rows.append(dict(domain=dom, brain=key, model=adj, beta=beta, se=se, p=pv,
                                 std_beta=beta * d.dN.std() / d.dS.std(),
                                 n_obs=len(d), n_patients=d.patient.nunique(),
                                 converged=r.converged))
    out = pd.DataFrame(rows)
    for adj in ["adjusted", "unadjusted"]:
        msk = out.model == adj
        out.loc[msk, "p_FDR"] = bh(out.loc[msk, "p"])
    return out


# ---------------------------------------------------------------- B2. frequency-specific coupling
def run_B2(suffix=""):
    rows = []
    off = post[post.condition == "OFF"].set_index(["patient", "visit"])
    stim = post[post.condition.isin(["LFS", "HFS", "VFS"])].copy()
    for dom in DOMAINS_BC:
        for key, col in BRAIN.items():
            bc = ("S4_pp" if key == "S4_pp" else col) + suffix
            recs = []
            for _, s in stim.iterrows():
                k = (s.patient, s.visit)
                if k not in off.index:
                    continue
                o = off.loc[k]
                if pd.isna(s[dom]) or pd.isna(o[dom]):
                    continue
                recs.append(dict(patient=s.patient, visit=s.visit, condition=s.condition,
                                 dS=s[dom] - o[dom], dN=s[bc] - o[bc]))
            d = pd.DataFrame(recs)
            d["visit"] = pd.Categorical(d["visit"], VISITS)
            r = fit(f"dS ~ dN * {CF} + {VF}", d)
            fe, V = r.fe_params, r.cov_params().loc[r.fe_params.index, r.fe_params.index]
            inter = [f"dN:{cname(l)}" for l in ["LFS", "VFS"]]
            bi = fe[inter].values; Vi = V.loc[inter, inter].values
            from scipy.stats import chi2
            wald = float(bi @ np.linalg.solve(Vi, bi)); p_int = chi2.sf(wald, 2)
            row = dict(domain=dom, brain=key, wald_chi2=wald, p_interaction=p_int,
                       n_obs=len(d), n_patients=d.patient.nunique(), converged=r.converged)
            for l in ["HFS", "LFS", "VFS"]:
                L = pd.Series(0.0, index=fe.index); L["dN"] = 1
                if l != "HFS":
                    L[f"dN:{cname(l)}"] = 1
                est = float(L @ fe); se = float(np.sqrt(L @ V @ L))
                row[f"slope_{l}"] = est; row[f"p_{l}"] = 2 * norm.sf(abs(est / se))
            rows.append(row)
    out = pd.DataFrame(rows); out["p_interaction_FDR"] = bh(out.p_interaction)
    return out


# ---------------------------------------------------------------- C. longitudinal coupling
def run_C(suffix=""):
    rows = []
    pre = df[df.condition == "PREOP"].set_index("patient")
    hfs = post[post.condition == "HFS"].copy()
    for dom in DOMAINS_BC:
        for key, col in BRAIN.items():
            bc = ("S4_pp" if key == "S4_pp" else col) + suffix
            recs = []
            for _, s in hfs.iterrows():
                if s.patient not in pre.index:
                    continue
                p0 = pre.loc[s.patient]
                if pd.isna(s[dom]) or pd.isna(p0[dom]):
                    continue
                recs.append(dict(patient=s.patient, visit=s.visit,
                                 dS=s[dom] - p0[dom], dN=s[bc] - p0[bc]))
            d = pd.DataFrame(recs)
            d["visit"] = pd.Categorical(d["visit"], VISITS)
            r = fit(f"dS ~ dN + {VF}", d)
            beta, se, pv = r.fe_params["dN"], r.bse["dN"], r.pvalues["dN"]
            rows.append(dict(domain=dom, brain=key, beta=beta, se=se, p=pv,
                             std_beta=beta * d.dN.std() / d.dS.std(),
                             n_obs=len(d), n_patients=d.patient.nunique(),
                             converged=r.converged))
    out = pd.DataFrame(rows); out["p_FDR"] = bh(out.p)
    return out


resB = run_B(); resB.to_csv(OUT / "B_within_visit_coupling.csv", index=False)
resBs = run_B("_alt"); resBs.to_csv(OUT / "B_within_visit_coupling_v2inverted.csv", index=False)
resB2 = run_B2(); resB2.to_csv(OUT / "B2_frequency_specific_coupling.csv", index=False)
resB2s = run_B2("_alt"); resB2s.to_csv(OUT / "B2_frequency_specific_coupling_v2inverted.csv", index=False)
resC = run_C(); resC.to_csv(OUT / "C_longitudinal_coupling_HFS.csv", index=False)
resCs = run_C("_alt"); resCs.to_csv(OUT / "C_longitudinal_coupling_HFS_v2inverted.csv", index=False)

cols = ["domain", "brain", "beta", "std_beta", "p", "p_FDR", "n_obs", "n_patients", "converged"]
print("\n[B] Within-visit coupling, condition-adjusted (primary)\n",
      resB[resB.model == "adjusted"][cols].round(4).to_string(index=False))
print("\n[B] Within-visit coupling, unadjusted\n",
      resB[resB.model == "unadjusted"][cols].round(4).to_string(index=False))
print("\n[B] v2-inverted, adjusted\n",
      resBs[resBs.model == "adjusted"][cols].round(4).to_string(index=False))
c2 = ["domain", "brain", "slope_LFS", "p_LFS", "slope_HFS", "p_HFS", "slope_VFS", "p_VFS",
      "p_interaction", "p_interaction_FDR", "converged"]
print("\n[B2] Frequency-specific slopes (does coupling differ by frequency?)\n",
      resB2[c2].round(4).to_string(index=False))
print("\n[B2] v2-inverted\n", resB2s[c2].round(4).to_string(index=False))
print("\n[C] Longitudinal coupling, HFS vs preop\n", resC[cols].round(4).to_string(index=False))
print("\n[C] v2-inverted\n", resCs[cols].round(4).to_string(index=False))
print(f"\nAll outputs written to {OUT}")
