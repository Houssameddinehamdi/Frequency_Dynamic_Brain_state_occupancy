"""
Shared statistical helpers.

Linear mixed models with a random intercept per patient (statsmodels MixedLM, REML),
asymptotic Dunnett adjustment for the three contrasts against HFS, Holm adjustment for
the pairwise contrasts among the non-HFS conditions and Benjamini-Hochberg FDR.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import multivariate_normal, norm

CONDITIONS = ["OFF", "LFS", "HFS", "VFS"]
VISITS = ["1m", "3m", "6m", "12m"]
CF = "C(condition, Treatment('HFS'))"
VF = "C(visit, Treatment('1m'))"


def cname(level):
    """Name of the fixed-effect coefficient for `level` versus HFS."""
    return f"{CF}[T.{level}]"


def fit(formula, data, reml=True):
    model = smf.mixedlm(formula, data, groups=data["patient"])
    try:
        return model.fit(reml=reml, method="powell")
    except Exception:
        return model.fit(reml=reml, method="lbfgs")


def _box_prob(R, c):
    """P(|Z_i| < c for all i) for Z ~ N(0, R)."""
    k = R.shape[0]
    return multivariate_normal.cdf(np.full(k, c), mean=np.zeros(k), cov=R, allow_singular=True,
                                   maxpts=2_000_000, abseps=1e-7, releps=1e-7, lower_limit=np.full(k, -c))


def dunnett(res, levels=("VFS", "LFS", "OFF")):
    """Asymptotic Dunnett-adjusted P values and simultaneous 95% CIs (contrasts vs HFS)."""
    names = [cname(l) for l in levels]
    b = res.fe_params[names].values
    V = res.cov_params().loc[names, names].values
    se = np.sqrt(np.diag(V))
    z = b / se
    R = V / np.outer(se, se)
    p = np.clip([1 - _box_prob(R, abs(zi)) for zi in z], 0, 1)
    lo, hi = 1.5, 4.0
    for _ in range(60):
        c = (lo + hi) / 2
        lo, hi = (c, hi) if _box_prob(R, c) < 0.95 else (lo, c)
    crit = (lo + hi) / 2
    return pd.DataFrame(dict(contrast=[f"{l} - HFS" for l in levels], level=list(levels),
                             beta=b, se=se, z=z, p_raw=2 * norm.sf(np.abs(z)), p_adj=p,
                             ci_lo=b - crit * se, ci_hi=b + crit * se, crit_z=crit))


def holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    adj = np.empty(n); running = 0.0
    for i, k in enumerate(order):
        running = max(running, (n - i) * p[k]); adj[k] = min(1.0, running)
    return adj


def bh(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    adj = np.empty(n); running = 1.0
    for i in range(n - 1, -1, -1):
        k = order[i]; running = min(running, p[k] * n / (i + 1)); adj[k] = running
    return adj


def pairwise_holm(res, pairs=(("VFS", "LFS"), ("VFS", "OFF"), ("LFS", "OFF"))):
    """Contrasts among non-HFS conditions, Holm-adjusted."""
    fe = res.fe_params; V = res.cov_params().loc[fe.index, fe.index]
    rows = []
    for a, b in pairs:
        L = pd.Series(0.0, index=fe.index)
        if a != "HFS": L[cname(a)] = 1
        if b != "HFS": L[cname(b)] = -1
        est = float(L @ fe); se = float(np.sqrt(L @ V @ L))
        rows.append(dict(contrast=f"{a} - {b}", beta=est, se=se, z=est / se,
                         p_raw=2 * norm.sf(abs(est / se))))
    out = pd.DataFrame(rows)
    out["p_adj"] = holm(out.p_raw)
    return out


def condition_model(df, outcome):
    """outcome ~ condition (ref HFS) + visit (ref 1 month) + (1 | patient)."""
    d = df[df.condition.isin(CONDITIONS)].dropna(subset=[outcome]).copy()
    d["visit"] = pd.Categorical(d["visit"], VISITS)
    res = fit(f"{outcome} ~ {CF} + {VF}", d)
    return res, d


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
