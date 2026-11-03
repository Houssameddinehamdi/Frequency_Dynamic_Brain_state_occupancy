# Frequency-dependent dynamic brain-state occupancy under subthalamic DBS

Analysis code for a 12-month within-patient study of dynamic brain-state occupancy under four
subthalamic deep brain stimulation (STN-DBS) conditions in Parkinson's disease: DBS-OFF,
60 Hz (LFS), 130 Hz (HFS) and alternating 130/60 Hz (VFS).

The code covers the statistics and every data figure: condition effects on dynamic-state
measures (linear mixed models with Dunnett and Holm adjustment), the eigenvector-sign
sensitivity analysis, the per-visit comparisons, the clinical analyses and the figure panels.
Brain states are defined with Weighted Eigenvector Dynamics Analysis (WEiDA), which combines
the two leading eigenvectors of the BOLD phase-coherence matrix weighted by their eigenvalues.

## Try it in two minutes (no data needed)

```bash
git clone https://github.com/Houssameddinehamdi/Frequency_Dynamic_Brain_state_occupancy.git
cd Frequency_Dynamic_Brain_state_occupancy
pip install -r requirements.txt
bash run_demo.sh
```

`run_demo.sh` writes a **synthetic** data set (`demo_data/`, random numbers with the same file
layout as the real data), then runs all statistics and draws all figure panels into
`demo_outputs/`. The synthetic values have no scientific meaning; they only show that the
pipeline runs end to end.

## Run with the real data

```bash
bash run_all.sh /path/to/data_folder outputs
```

`/path/to/data_folder` must contain the session-level results and the clinical scores in the
layout described in [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md):

```
data_folder/
├── pathA_results/
│   ├── 01_session_metrics.csv
│   ├── 04_centroids.csv
│   ├── 05_hc_occupancy.csv
│   └── 13_fig5B_transition_matrices.csv
└── clinical_processed/
    └── UPDRS_{III,bradykinesia,rigidity,PIGD,tremor}_processed.tsv
```

Outputs:

| File / folder | Content |
|---|---|
| `outputs/tables/table3_mixed_models.csv` | contrasts vs HFS (Dunnett), contrasts among DBS-OFF, LFS and VFS (Holm), condition × visit test |
| `outputs/tables/table3_sensitivity_v2.csv` | the same contrasts after inverting the sign of the second eigenvector (v2) |
| `outputs/tables/table4_per_visit_hfs_vs_vfs.csv` | per-visit HFS vs VFS (paired t-tests, BH-FDR) |
| `outputs/tables/condition_means.csv` | condition means of each measure |
| `outputs/clinical_analysis/` | inter-rater ICC, clinical response by symptom domain, brain–symptom coupling |
| `outputs/figures/` | every data panel (SVG, PDF, 300-dpi PNG) |
| `outputs/figures/stats/fig2c_mann_whitney.csv` | healthy-control vs presurgical occupancy tests |

Single steps:

```bash
python scripts/primary_models.py   --base DATA --out OUT
python scripts/clinical_analysis.py --base DATA --out OUT
python scripts/make_figures.py     --base DATA --out OUT/figures --clin OUT/clinical_analysis [--only fig3a fig4a]
```

## Data

* **open-DBS cohort** (14 patients; presurgical and 1, 3, 6 and 12-month scans under DBS-OFF,
  60 Hz, 130 Hz and 130/60 Hz stimulation): OpenNeuro, https://doi.org/10.18112/openneuro.ds005849.v2.0.1
* **State-centroid training data and healthy-control reference**: not public. Access may be
  requested from the authors, subject to ethics approval.

No participant data are included in this repository.

## Upstream processing

From preprocessed fMRI to `pathA_results/` (regional time series, WEiDA features, state
assignment with fixed centroids, session metrics): see [`pipeline/`](pipeline/).

## Requirements

Python ≥ 3.9 with numpy, pandas, scipy, statsmodels and matplotlib (`requirements.txt`).
nilearn is needed only for the optional brain maps (`--atlas aal`).
Tested with Python 3.10 (numpy 1.26, pandas 2.2, scipy 1.15, statsmodels 0.15) and Python 3.13.

## Statistical notes

* Mixed models: `outcome ~ condition (reference HFS) + visit (reference 1 month) + (1 | patient)`,
  REML, statsmodels `MixedLM`.
* Contrasts against HFS: asymptotic Dunnett adjustment from the multivariate normal distribution
  of the three estimates, with simultaneous 95% confidence intervals. Contrasts among DBS-OFF,
  LFS and VFS: Holm. Static functional connectivity: Holm across all six condition pairs.
  Families of exploratory tests: Benjamini–Hochberg FDR.
* Condition × visit interaction: likelihood-ratio test of maximum-likelihood fits.
* The healthy-control vs presurgical comparison involves groups scanned at different sites;
  those Mann–Whitney tests are descriptive.

## License

MIT (see `LICENSE`).
