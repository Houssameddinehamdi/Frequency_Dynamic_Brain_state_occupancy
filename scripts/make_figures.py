#!/usr/bin/env python3
"""
make_figures.py - draws every data panel of the manuscript from the exported results.

Each panel is saved separately (SVG, PDF and 300-dpi PNG) so that panels can be assembled
in any layout tool. All values come from the input files; nothing is typed in by hand.

Usage
    python scripts/make_figures.py --base <data folder> --out <figure folder>
    python scripts/make_figures.py --base <data folder> --out <figure folder> --only fig2b fig3a
    python scripts/make_figures.py ... --atlas aal       # adds the Fig. 2A brain maps (needs nilearn + internet)

Inputs (inside --base)
    pathA_results/01_session_metrics.csv            session-level metrics
    pathA_results/04_centroids.csv                  4 state centroids x 93 regions        (Fig. 2A)
    pathA_results/05_hc_occupancy.csv               occupancy of each healthy control       (Fig. 2B-C)
    pathA_results/13_fig5B_transition_matrices.csv  transition probabilities                (Fig. 5B)
    clinical_processed/UPDRS_<domain>_processed.tsv clinical scores                         (Fig. 6, Suppl. Fig. 1)
    clinical_analysis/A_*.csv, B_*.csv              output of clinical_analysis.py          (Fig. 6)
                                                    (looked up in --clin, then <out>/.., then <base>)
Panels
    fig1   illustrative elements for the schematic (not data)
    fig2a  state centroids (bars; brain maps with --atlas)
    fig2b  mean occupancy, HC vs open-DBS preop (stacked bars linked by ribbons)
    fig2c  occupancy per participant with Mann-Whitney U tests (BH-FDR across states)
    fig3a  occupancy by condition (violins; State 4 Dunnett contrasts vs HFS)
    fig3b  d_HC by condition and visit;  fig3c  d_HC contrasts vs HFS
    fig4a  State 4 residence time (violins);  fig4b  contrasts;  fig4c  primary vs v2-inverted
    fig5a  static FC (violins);  fig5b  transition-probability differences vs HFS (descriptive)
    fig6a  clinical scores by domain (violins);  fig6b  brain-symptom coupling
    supp1  concordance between the motor-optimal and the template-closest condition
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.stats import mannwhitneyu

from npj1_stats import CONDITIONS as COND, VISITS, condition_model, dunnett, pairwise_holm, bh, stars

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ style
MM = 1 / 25.4
COND_LABEL = {"OFF": "DBS-OFF", "LFS": "LFS", "HFS": "HFS", "VFS": "VFS"}
CCOL = {"OFF": "#5C5C5C", "LFS": "#1A9E77", "HFS": "#D55E00", "VFS": "#3B6FB6"}
SCOL = {"S1": "#CC6677", "S2": "#B8860B", "S3": "#2A9D8F", "S4": "#7B3294"}
POS, NEG = "#B2182B", "#2166AC"
INK, MUTED, GRID = "#1A1A1A", "#6B6B6B", "#D9D9D9"
VISIT_X = {"1m": 1, "3m": 3, "6m": 6, "12m": 12}
DHC = r"$d_{\mathrm{HC}}$"
STATES = ["S1", "S2", "S3", "S4"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "svg.fonttype": "path", "pdf.fonttype": 42, "savefig.dpi": 300,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  saved {name}")


def fmt_p(p):
    return "P < 0.001" if p < 0.001 else f"P = {p:.3f}"


# ------------------------------------------------------------------ data
def load_sessions():
    d = pd.read_csv(RES / "01_session_metrics.csv").rename(columns={"subject": "patient", "timepoint": "visit"})
    for s in STATES:
        d[f"{s}_pct"] = d[s] * 100
        if f"{s}_alt" in d:
            d[f"{s}_pct_alt"] = d[f"{s}_alt"] * 100
    return d


def subject_means(d, col):
    """Mean over visits within each patient and condition (one value per patient)."""
    return d[d.condition.isin(COND)].groupby(["patient", "condition"])[col].mean().unstack()[COND]


def contrasts_vs_hfs(d, col):
    res, _ = condition_model(d, col)
    return dunnett(res), pairwise_holm(res)


def hc_and_preop():
    hc = pd.read_csv(RES / "05_hc_occupancy.csv")
    cols = [c for c in hc.columns if c.upper() in STATES]
    if len(cols) != 4:
        sys.exit(f"05_hc_occupancy.csv needs columns S1..S4; found {list(hc.columns)}")
    hc = hc[cols].rename(columns=str.upper)
    if hc.values.max() <= 1.0:
        hc = hc * 100
    pre = load_sessions()
    pre = pre[pre.condition == "PREOP"][STATES] * 100
    return hc.reset_index(drop=True), pre.reset_index(drop=True)


# ------------------------------------------------------------------ reusable panels
def paired_violin(ax, wide, ylabel, title=None):
    """Violin (density over the observed range) + one point per patient + grey line joining
    the same patient across conditions + median (white dot) and interquartile range (bar)."""
    x = np.arange(len(COND))
    pats = list(wide.index)
    jit = dict(zip(pats, np.linspace(-0.08, 0.08, len(pats)) if len(pats) > 1 else [0.0]))
    for p, row in wide.iterrows():
        ax.plot(x + jit[p], row.values, color=GRID, lw=0.5, zorder=1)
    for i, c in enumerate(COND):
        v = wide[c].dropna().values
        if len(v) > 1 and np.ptp(v) > 0:
            vp = ax.violinplot(v, positions=[i], widths=0.72, showextrema=False, showmedians=False)
            for b in vp["bodies"]:
                b.set_facecolor(CCOL[c]); b.set_alpha(0.18); b.set_edgecolor(CCOL[c]); b.set_linewidth(0.6)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([i + 0.2, i + 0.2], [q1, q3], color=CCOL[c], lw=2.2, solid_capstyle="butt", zorder=4)
        ax.scatter([i + 0.2], [med], s=10, color="white", edgecolor=CCOL[c], lw=0.8, zorder=5)
        xs = [i + jit[p] for p in wide[c].dropna().index]
        ax.scatter(xs, v, s=9, color=CCOL[c], edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([COND_LABEL[c] for c in COND])
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlim(-0.6, len(COND) - 0.4)


def brackets_vs_hfs(ax, wide, pvals):
    """Brackets from HFS to each listed condition with Dunnett-adjusted significance marks."""
    top = np.nanmax(wide.values); rng = np.nanmax(wide.values) - np.nanmin(wide.values)
    h = rng * 0.07; y = top + h * 0.6
    ih = COND.index("HFS")
    for lvl in ["VFS", "LFS", "OFF"]:
        if lvl not in pvals:
            continue
        i = COND.index(lvl)
        ax.plot([ih, ih, i, i], [y, y + h * 0.4, y + h * 0.4, y], color=INK, lw=0.6)
        ax.text((ih + i) / 2, y + h * 0.45, stars(pvals[lvl]), ha="center", va="bottom", fontsize=6.5)
        y += h * 1.3
    ax.set_ylim(top=y + h * 0.3)


def forest(ax, tab, xlabel):
    tab = tab.reset_index(drop=True)
    ys = np.arange(len(tab))[::-1]
    for y, r in zip(ys, tab.itertuples()):
        col = CCOL.get(r.level, INK)
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=1.2, solid_capstyle="round")
        ax.scatter([r.beta], [y], s=22, color=col, edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(1.02, y, f"{r.beta:.4f}".replace("-", "−") + f"   {fmt_p(r.p_adj)}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=6, color=INK)
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(ys); ax.set_yticklabels([f"{COND_LABEL[l]} vs HFS" for l in tab.level])
    ax.set_xlabel(xlabel)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)


# ================================================================== FIGURE 1 elements (illustrative)
def fig1():
    from scipy.signal import hilbert
    rng = np.random.default_rng(3)
    t = np.arange(0, 120)
    sig = [np.sin(2 * np.pi * 0.05 * t + ph) + 0.35 * rng.standard_normal(t.size) for ph in (0, 0.6, 2.5)]
    cols = ["#444444", "#777777", "#AAAAAA"]
    for name, f in [("bold", lambda s: s), ("phase", lambda s: np.angle(hilbert(s)))]:
        fig, ax = plt.subplots(figsize=(45 * MM, 22 * MM))
        for k, s in enumerate(sig):
            ax.plot(t, f(s) + (3 if name == "bold" else 7) * k, color=cols[k], lw=0.8)
        ax.axvline(60, color=CCOL["HFS"], lw=0.8, ls=(0, (2, 2))); ax.set_axis_off()
        save(fig, f"fig1_elem_{name}")
    n = 40
    theta = np.r_[rng.normal(0, 0.3, n // 2), rng.normal(np.pi * 0.8, 0.3, n - n // 2)]
    pc = np.cos(theta[:, None] - theta[None, :])
    fig, ax = plt.subplots(figsize=(28 * MM, 28 * MM))
    ax.imshow(pc, cmap="RdBu_r", vmin=-1, vmax=1); ax.set_xticks([]); ax.set_yticks([])
    save(fig, "fig1_elem_pcmatrix")
    w, v = np.linalg.eigh(pc)
    for name, vec in [("v1", v[:, -1]), ("v2", v[:, -2])]:
        fig, ax = plt.subplots(figsize=(8 * MM, 28 * MM))
        ax.barh(np.arange(n), vec, color=[POS if a > 0 else NEG for a in vec], height=0.8)
        ax.axvline(0, color=INK, lw=0.4); ax.invert_yaxis(); ax.set_axis_off()
        save(fig, f"fig1_elem_{name}")
    fig, ax = plt.subplots(figsize=(30 * MM, 28 * MM))
    for k, (s, c) in enumerate(SCOL.items()):
        mu = np.array([np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)]) * 2
        pts = mu + rng.normal(0, 0.55, (60, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=3, color=c, alpha=0.6, linewidth=0)
        ax.scatter(*mu, s=40, color=c, edgecolor=INK, linewidth=0.8, marker="X", zorder=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    save(fig, "fig1_elem_kmeans")


# ================================================================== FIGURE 2
def fig2a(atlas=None):
    cen = pd.read_csv(RES / "04_centroids.csv")
    num = cen.select_dtypes("number")
    if num.shape[0] == 4:
        C = num.values; rois = list(num.columns)
    elif num.shape[1] == 4:
        C = num.values.T
        nn = cen.select_dtypes(exclude="number")
        rois = list(nn.iloc[:, 0]) if nn.shape[1] else [f"R{i + 1}" for i in range(C.shape[1])]
    else:
        sys.exit(f"04_centroids.csv has shape {cen.shape}; expected 4 states x regions")
    fig, axs = plt.subplots(1, 4, figsize=(180 * MM, 50 * MM))
    for k, ax in enumerate(axs):
        v = C[k]; idx = np.argsort(-np.abs(v))[:10][::-1]
        ax.barh(range(10), v[idx], color=[POS if a > 0 else NEG for a in v[idx]], height=0.7)
        ax.set_yticks(range(10)); ax.set_yticklabels([str(rois[i]) for i in idx], fontsize=5.5)
        ax.axvline(0, color=INK, lw=0.5)
        ax.set_title(f"State {k + 1}", loc="left", fontweight="bold", color=SCOL[f"S{k + 1}"])
        ax.set_xlabel("Centroid loading")
    fig.tight_layout(); save(fig, "fig2a_top_regions")
    vmax = np.abs(C).max()
    fig, ax = plt.subplots(figsize=(30 * MM, 4 * MM))
    fig.colorbar(matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(-vmax, vmax), "RdBu_r"),
                 cax=ax, orientation="horizontal", label="Centroid loading")
    save(fig, "fig2a_colorbar")
    if atlas != "aal":
        print("  brain maps skipped (use --atlas aal to draw them)")
        return
    try:
        from nilearn import datasets, image, plotting, surface
        aal = datasets.fetch_atlas_aal(version="SPM12")
    except Exception as e:
        print(f"  brain maps skipped: {e}"); return
    name2val = dict(zip([str(l) for l in aal.labels], [int(i) for i in aal.indices]))
    keep = [j for j, r in enumerate(rois) if str(r) in name2val]
    print(f"  {len(keep)} regions matched to AAL by name (others are not drawn on the brain)")
    img = image.load_img(aal.maps); lab = np.asarray(img.get_fdata()).astype(int)
    fsavg = datasets.fetch_surf_fsaverage("fsaverage5")
    coords_all, ids = plotting.find_parcellation_cut_coords(img, return_label_names=True)
    id2c = dict(zip([int(i) for i in ids], coords_all))
    labels = [name2val[str(rois[j])] for j in keep]
    coords = np.array([id2c[L] for L in labels])
    for k in range(C.shape[0]):
        vol = np.zeros(lab.shape)
        for j, L in zip(keep, labels):
            vol[lab == L] = C[k, j]
        simg = image.new_img_like(img, vol)
        fig, axs = plt.subplots(1, 4, figsize=(170 * MM, 42 * MM), subplot_kw={"projection": "3d"})
        for ax, (hemi, view) in zip(axs, [("left", "lateral"), ("left", "medial"), ("right", "medial"), ("right", "lateral")]):
            tex = surface.vol_to_surf(simg, fsavg[f"pial_{hemi}"], interpolation="nearest")
            plotting.plot_surf_stat_map(fsavg[f"infl_{hemi}"], tex, hemi=hemi, view=view, cmap="RdBu_r", vmax=vmax,
                                        colorbar=False, bg_map=fsavg[f"sulc_{hemi}"], axes=ax, figure=fig)
        save(fig, f"fig2a_surface_state{k + 1}")
        ck = C[k, keep]
        disp = plotting.plot_markers(ck, coords, node_cmap="RdBu_r", node_vmin=-vmax, node_vmax=vmax,
                                     node_size=np.abs(ck) / vmax * 40 + 4, display_mode="lzr", colorbar=False,
                                     title=f"State {k + 1}")
        for ext in ("svg", "pdf", "png"):
            disp.savefig(str(OUT / f"fig2a_glass_state{k + 1}.{ext}"), dpi=300)
        disp.close(); print(f"  saved fig2a_glass_state{k + 1}")


def fig2b():
    """Stacked bars linked by ribbons that join the same state in the two groups.
    Ribbons show the change in each state's share; they do not represent transitions."""
    hc, pre = hc_and_preop()
    g = pd.DataFrame([hc.mean(), pre.mean()], index=["HC", "open-DBS\npreop"])[STATES]
    g = g.div(g.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(70 * MM, 50 * MM))
    w = 0.28; xs = [0, 1]
    bottoms = np.zeros((2, 4)); tops = np.zeros((2, 4))
    for gi in range(2):
        b = 0.0
        for si, s in enumerate(STATES):
            bottoms[gi, si] = b; b += g.iloc[gi][s]; tops[gi, si] = b
    for si, s in enumerate(STATES):
        poly = Polygon([[xs[0] + w, bottoms[0, si]], [xs[1] - w, bottoms[1, si]],
                        [xs[1] - w, tops[1, si]], [xs[0] + w, tops[0, si]]],
                       closed=True, facecolor=SCOL[s], alpha=0.25, edgecolor="none")
        ax.add_patch(poly)
        for gi in range(2):
            v = g.iloc[gi][s]
            ax.bar(xs[gi], v, bottom=bottoms[gi, si], width=2 * w, color=SCOL[s], edgecolor="white", linewidth=0.8,
                   label=f"State {s[1]}" if gi == 0 else None)
            ax.text(xs[gi], bottoms[gi, si] + v / 2, f"{v:.1f}", ha="center", va="center", fontsize=6, color="white")
    ax.set_xticks(xs); ax.set_xticklabels([f"HC\n(n = {len(hc)})", f"open-DBS preop\n(n = {len(pre)})"])
    ax.set_ylabel("Occupancy (%)"); ax.set_ylim(0, 100); ax.set_xlim(-0.5, 1.5)
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1)
    save(fig, "fig2b_cohort_occupancy_ribbons")


def fig2c():
    """Per-participant occupancy; two-sided Mann-Whitney U per state, BH-FDR across the four states.
    The two groups come from different sites, so the comparison is descriptive."""
    hc, pre = hc_and_preop()
    rows = []
    for s in STATES:
        u, p = mannwhitneyu(hc[s], pre[s], alternative="two-sided")
        rows.append(dict(state=s, n_HC=len(hc), n_preop=len(pre), median_HC=hc[s].median(),
                         median_preop=pre[s].median(), U=u, p_raw=p))
    st = pd.DataFrame(rows); st["p_FDR"] = bh(st.p_raw)
    TAB.mkdir(parents=True, exist_ok=True); st.to_csv(TAB / "fig2c_mann_whitney.csv", index=False)
    print(st.round(4).to_string(index=False))
    fig, axs = plt.subplots(1, 4, figsize=(90 * MM, 42 * MM), sharey=True)
    ymax = max(hc.values.max(), pre.values.max())
    for ax, s, p in zip(axs, STATES, st.p_FDR):
        for i, vals in enumerate([hc[s].values, pre[s].values]):
            jit = np.random.default_rng(i).uniform(-0.12, 0.12, len(vals))
            ax.scatter(i + jit, vals, s=8, color=SCOL[s] if i else "#9A9A9A", edgecolor="white", linewidth=0.3)
            ax.plot([i - 0.22, i + 0.22], [np.median(vals)] * 2, color=INK, lw=1)
        y = ymax * 1.06
        ax.plot([0, 0, 1, 1], [y, y * 1.02, y * 1.02, y], color=INK, lw=0.6)
        ax.text(0.5, y * 1.03, stars(p), ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["HC", "open-DBS\npreop"])
        ax.set_title(f"State {s[1]}", loc="left", fontweight="bold"); ax.set_xlim(-0.5, 1.5)
    axs[0].set_ylabel("Occupancy (%)"); axs[0].set_ylim(top=ymax * 1.15)
    save(fig, "fig2c_hc_vs_preop")


# ================================================================== FIGURE 3
def fig3a():
    d = load_sessions()
    D, _ = contrasts_vs_hfs(d, "S4_pct")
    print("  State 4 occupancy contrasts vs HFS:\n", D[["contrast", "beta", "p_adj"]].round(4).to_string(index=False))
    fig, axs = plt.subplots(1, 4, figsize=(180 * MM, 48 * MM))
    for ax, s in zip(axs, STATES):
        w = subject_means(d, f"{s}_pct")
        paired_violin(ax, w, "Occupancy (%)" if s == "S1" else "", title=f"State {s[1]}")
        ax.set_xticklabels(COND)
        if s == "S4":
            brackets_vs_hfs(ax, w, dict(zip(D.level, D.p_adj)))
    fig.tight_layout(); save(fig, "fig3a_occupancy_by_condition")


def fig3b():
    d = load_sessions()
    fig, ax = plt.subplots(figsize=(75 * MM, 50 * MM))
    off = {"OFF": -0.3, "LFS": -0.1, "HFS": 0.1, "VFS": 0.3}
    for c in COND:
        sub = d[d.condition == c].groupby("visit")["dist_HC"].agg(["mean", "sem"]).reindex(VISITS)
        x = np.array([VISIT_X[v] for v in VISITS]) + off[c]
        ax.errorbar(x, sub["mean"], yerr=sub["sem"], color=CCOL[c], lw=1.2, marker="o", ms=3.5,
                    mec="white", mew=0.5, capsize=1.5, elinewidth=0.8, label=COND_LABEL[c])
    ax.set_xticks([1, 3, 6, 12]); ax.set_xlabel("Months after surgery"); ax.set_ylabel(f"{DHC} (mean ± s.e.m.)")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.5)
    save(fig, "fig3b_dHC_trajectories")


def fig3c():
    D, _ = contrasts_vs_hfs(load_sessions(), "dist_HC")
    fig, ax = plt.subplots(figsize=(60 * MM, 28 * MM))
    forest(ax, D, f"Difference in {DHC} vs HFS\n(simultaneous 95% CI)")
    save(fig, "fig3c_dHC_contrasts")


# ================================================================== FIGURE 4
def fig4a():
    d = load_sessions()
    w = subject_means(d, "S4_residence_TR")
    D, _ = contrasts_vs_hfs(d, "S4_residence_TR")
    fig, ax = plt.subplots(figsize=(62 * MM, 52 * MM))
    paired_violin(ax, w, "State 4 residence time (TR)")
    brackets_vs_hfs(ax, w, dict(zip(D.level, D.p_adj)))
    save(fig, "fig4a_residence")


def fig4b():
    D, _ = contrasts_vs_hfs(load_sessions(), "S4_residence_TR")
    fig, ax = plt.subplots(figsize=(60 * MM, 28 * MM))
    forest(ax, D, "Difference in residence time vs HFS (TR)")
    save(fig, "fig4b_residence_contrasts")


def fig4c():
    d = load_sessions()
    if "dist_HC_alt" not in d:
        print("  no *_alt columns: Fig. 4C skipped"); return
    fig, axs = plt.subplots(1, 2, figsize=(120 * MM, 30 * MM))
    for ax, (col, lab) in zip(axs, [("dist_HC", f"Difference in {DHC} vs HFS"),
                                    ("S4_residence_TR", "Difference in residence vs HFS (TR)")]):
        P, _ = contrasts_vs_hfs(d, col)
        A, _ = contrasts_vs_hfs(d.drop(columns=[col]).rename(columns={f"{col}_alt": col}), col)
        ys = np.arange(3)[::-1]
        for y, rp, ra in zip(ys, P.itertuples(), A.itertuples()):
            ax.plot([rp.ci_lo, rp.ci_hi], [y + 0.12] * 2, color=CCOL[rp.level], lw=1.1)
            ax.scatter([rp.beta], [y + 0.12], s=18, color=CCOL[rp.level], zorder=3)
            ax.plot([ra.ci_lo, ra.ci_hi], [y - 0.12] * 2, color=CCOL[ra.level], lw=1.1, ls=(0, (2, 1.5)))
            ax.scatter([ra.beta], [y - 0.12], s=18, facecolor="white", edgecolor=CCOL[ra.level], lw=1, zorder=3)
            ax.text(1.02, y, f"{fmt_p(rp.p_adj)} | {fmt_p(ra.p_adj)}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=5.5)
        ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        ax.set_yticks(ys); ax.set_yticklabels([f"{COND_LABEL[l]} vs HFS" for l in P.level])
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0); ax.set_xlabel(lab)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(handles=[Line2D([], [], color=INK, marker="o", lw=1, label="Primary"),
                        Line2D([], [], color=INK, marker="o", mfc="white", lw=1, ls=(0, (2, 1.5)), label="v2 inverted")],
               frameon=False, loc="upper center", ncol=2)
    save(fig, "fig4c_sign_sensitivity")


# ================================================================== FIGURE 5
def fig5a():
    d = load_sessions()
    res, _ = condition_model(d, "static_fc")
    pairs = (("VFS", "HFS"), ("LFS", "HFS"), ("OFF", "HFS"), ("VFS", "LFS"), ("VFS", "OFF"), ("LFS", "OFF"))
    H = pairwise_holm(res, pairs=pairs)
    print(f"  static FC: smallest Holm-adjusted P across the six pairs = {H.p_adj.min():.3f}")
    fig, ax = plt.subplots(figsize=(62 * MM, 50 * MM))
    paired_violin(ax, subject_means(d, "static_fc"), "Static FC (mean |r|)")
    save(fig, "fig5a_static_fc")


def fig5b():
    """Group-mean transition matrix of each condition minus HFS (descriptive; no test)."""
    t = pd.read_csv(RES / "13_fig5B_transition_matrices.csv")
    if "version" in t.columns:
        t = t[t.version == "published"]
    low = {c.lower(): c for c in t.columns}
    cond = low["condition"]
    fr = next((low[k] for k in ("from", "from_state") if k in low), None)
    to = next((low[k] for k in ("to", "to_state") if k in low), None)
    pr = next((low[k] for k in ("prob", "probability", "p", "value") if k in low), None)
    M = {}
    for c, g in t.groupby(cond):
        m = np.zeros((4, 4)); n = np.zeros((4, 4))
        if fr and to and pr:                                   # long format
            for r in g.itertuples(index=False):
                i = int(str(getattr(r, fr)).strip("S")) - 1; j = int(str(getattr(r, to)).strip("S")) - 1
                m[i, j] += getattr(r, pr); n[i, j] += 1
        else:                                                  # wide format: one row per from-state
            num = g.select_dtypes("number").iloc[:, -4:].values
            rows = g[fr].astype(str).str.strip("S").astype(int).values - 1 if fr else np.tile(np.arange(4), len(g) // 4)
            for i, row in zip(rows, num):
                m[i] += row; n[i] += 1
        M[c] = m / np.maximum(n, 1)
    diffs = {c: M[c] - M["HFS"] for c in ["OFF", "LFS", "VFS"]}
    vmax = max(np.abs(v).max() for v in diffs.values())
    print(f"  colour scale ±{vmax:.3f}")
    fig, axs = plt.subplots(1, 3, figsize=(120 * MM, 40 * MM))
    for ax, c in zip(axs, ["OFF", "LFS", "VFS"]):
        im = ax.imshow(diffs[c], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{COND_LABEL[c]} − HFS", fontsize=6.5)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels([f"S{i}" for i in range(1, 5)]); ax.set_yticklabels([f"S{i}" for i in range(1, 5)])
        ax.set_xlabel("To state")
    axs[0].set_ylabel("From state")
    fig.colorbar(im, ax=axs, shrink=0.8, label="Δ transition probability\n(condition − HFS)")
    save(fig, "fig5b_transition_differences")


# ================================================================== FIGURE 6 (clinical)
DOMAINS = [("UPDRS_III", "UPDRS-III total"), ("bradykinesia", "Bradykinesia"), ("rigidity", "Rigidity"),
           ("PIGD", "PIGD (axial/gait)"), ("tremor", "Tremor")]
CODE = {"OFF": "OFF", "CL": "LFS", "CH": "HFS", "CV": "VFS"}


def load_clinical():
    rows = []
    for dom, _ in DOMAINS:
        t = pd.read_csv(CLIN / f"UPDRS_{dom.replace('UPDRS_', '')}_processed.tsv", sep="\t")
        for r in t.itertuples():
            c = str(r.condition)
            if c == "preop":
                continue
            v, code = c[:c.index("m") + 1], c[c.index("m") + 1:]
            if code in CODE:
                rows.append(dict(patient=r.participant_id, visit=v, condition=CODE[code], domain=dom, score=r.aver))
    return pd.DataFrame(rows)


def clin_results(name):
    for folder in [CLIN_RES, OUT.parent / "clinical_analysis", BASE / "clinical_analysis"]:
        if folder is not None and (Path(folder) / name).exists():
            return pd.read_csv(Path(folder) / name)
    sys.exit(f"{name} not found: run clinical_analysis.py first or pass --clin")


def fig6a():
    cl = load_clinical()
    A = clin_results("A_condition_effects_by_domain.csv")
    fig, axs = plt.subplots(1, 5, figsize=(180 * MM, 52 * MM))
    for ax, (dom, lab) in zip(axs, DOMAINS):
        w = cl[cl.domain == dom].groupby(["patient", "condition"]).score.mean().unstack()[COND]
        paired_violin(ax, w, "Score (mean over visits)" if dom == "UPDRS_III" else "", title=lab)
        ax.set_xticklabels(COND)
        a = A[(A.domain == dom) & (A.method == "Dunnett")]
        pv = {r.contrast.split(" ")[0]: r.p_adj for r in a.itertuples()}
        brackets_vs_hfs(ax, w, {k: pv[k] for k in ["VFS", "LFS"] if k in pv})
    fig.tight_layout(); save(fig, "fig6a_clinical_domains")


def fig6b():
    Bc = clin_results("B_within_visit_coupling.csv")
    ratio = Bc.std_beta / Bc.beta
    lo = (Bc.beta - 1.96 * Bc.se) * ratio; hi = (Bc.beta + 1.96 * Bc.se) * ratio
    Bc["lo"], Bc["hi"] = np.minimum(lo, hi), np.maximum(lo, hi)
    names = {"dist_HC": DHC, "S4_pp": "State 4 occupancy", "S4_residence_TR": "State 4 residence"}
    doms = ["rigidity", "bradykinesia", "PIGD", "tremor"]
    fig, axs = plt.subplots(1, 4, figsize=(180 * MM, 34 * MM), sharex=True)
    for ax, dom in zip(axs, doms):
        sub = Bc[Bc.domain == dom]
        for j, bm in enumerate(["dist_HC", "S4_pp", "S4_residence_TR"]):
            y = 2 - j
            for mdl, dy, filled in [("unadjusted", 0.14, True), ("adjusted", -0.14, False)]:
                r = sub[(sub.brain == bm) & (sub.model == mdl)].iloc[0]
                ax.plot([r.lo, r.hi], [y + dy] * 2, color=INK if filled else MUTED, lw=1, ls="-" if filled else (0, (2, 1.5)))
                ax.scatter([r.std_beta], [y + dy], s=16, color=INK if filled else "white",
                           edgecolor=INK if filled else MUTED, lw=0.9, zorder=3)
                if r.p_FDR < 0.05:
                    ax.text(r.hi, y + dy, " *", va="center", fontsize=7)
        ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        ax.set_yticks([2, 1, 0]); ax.set_yticklabels([names[b] for b in names] if dom == doms[0] else [])
        ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
        ax.set_title({"PIGD": "PIGD (axial/gait)"}.get(dom, dom.capitalize()), loc="left", fontweight="bold")
        ax.set_xlabel("Standardized β (95% CI)")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.legend(handles=[Line2D([], [], color=INK, marker="o", lw=1, label="Between-condition (unadjusted)"),
                        Line2D([], [], color=MUTED, marker="o", mfc="white", lw=1, ls=(0, (2, 1.5)),
                               label="Within-condition (adjusted)")],
               frameon=False, loc="upper center", ncol=2)
    fig.text(0.99, -0.02, "* FDR-adjusted P < 0.05. Negative β: a larger increase in the brain measure "
             "accompanied a larger improvement.", ha="right", fontsize=5.5, color=MUTED)
    save(fig, "fig6b_brain_symptom_coupling")


# ================================================================== SUPPLEMENTARY FIG. 1
def supp1():
    d = load_sessions()
    u = load_clinical(); u = u[u.domain == "UPDRS_III"]
    m = d[d.condition.isin(COND)].merge(u[["patient", "visit", "condition", "score"]], on=["patient", "visit", "condition"])
    rows = []
    for (p, v), g in m.groupby(["patient", "visit"]):
        if len(g) < 4 or g.score.isna().any():
            continue
        g = g.set_index("condition").loc[COND]          # ties: first in the order DBS-OFF, LFS, HFS, VFS
        rows.append(dict(patient=p, visit=v, motor=g.score.idxmin(), template=g.dist_HC.idxmin()))
    r = pd.DataFrame(rows); r["concordant"] = r.motor == r.template
    print(f"  concordant in {r.concordant.sum()} of {len(r)} patient-visits")
    fig, ax = plt.subplots(figsize=(70 * MM, 40 * MM))
    x = np.arange(4)
    ax.bar(x - 0.18, [np.sum(r.motor == c) for c in COND], 0.34, color=[CCOL[c] for c in COND], label="Lowest UPDRS-III")
    ax.bar(x + 0.18, [np.sum(r.template == c) for c in COND], 0.34, color="white",
           edgecolor=[CCOL[c] for c in COND], lw=1, hatch="////", label=f"Lowest {DHC}")
    ax.set_xticks(x); ax.set_xticklabels([COND_LABEL[c] for c in COND]); ax.set_ylabel("Patient-visits (n)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "supp1_concordance_counts")


# ================================================================== main
JOBS = ["fig1", "fig2a", "fig2b", "fig2c", "fig3a", "fig3b", "fig3c", "fig4a", "fig4b", "fig4c",
        "fig5a", "fig5b", "fig6a", "fig6b", "supp1"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="data folder (contains pathA_results/ and clinical_processed/)")
    ap.add_argument("--out", required=True, help="folder for the figure panels")
    ap.add_argument("--clin", default=None, help="folder with the clinical_analysis.py outputs")
    ap.add_argument("--atlas", default=None, help="'aal' to draw the Fig. 2A brain maps with nilearn's AAL atlas")
    ap.add_argument("--only", nargs="*", default=None, choices=JOBS)
    a = ap.parse_args()
    BASE = Path(a.base); RES = BASE / "pathA_results"; CLIN = BASE / "clinical_processed"
    OUT = Path(a.out); TAB = OUT / "stats"; CLIN_RES = Path(a.clin) if a.clin else None
    funcs = {k: globals()[k] for k in JOBS}
    funcs["fig2a"] = lambda: fig2a(a.atlas)
    import traceback
    failed = []
    for k in (a.only or JOBS):
        print(f"[{k}]")
        try:
            funcs[k]()
        except SystemExit as e:
            print(f"  {k} skipped: {e}"); failed.append(k)
        except Exception:
            traceback.print_exc(); failed.append(k)
    print(f"\nPanels written to {OUT}" + (f"  |  not drawn: {failed}" if failed else ""))
