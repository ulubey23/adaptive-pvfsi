# Adaptive PV-FSI

Analysis code for:

**"Adaptive PV-FSI: Defect-Specific Sub-Index Weighting and Irradiance-Normalized Severity Assessment for UAV-Based Photovoltaic Thermographic Inspection"**

M. Kabir, S. Ekici, T. C. Akinci, A. A. Martinez-Morales

This repository contains the full analysis pipeline used to produce every table and figure in the manuscript, built on top of the PV-FSI composite thermographic severity index and validated on the [DTU Drone Infrared Thermography Dataset](https://data.dtu.dk/) (open access, 640 acquisitions across seven PV module defect types).

## What this code does

Starting from the DTU dataset's raw radiometric thermograms and per-module I–V curves, the pipeline:

1. Extracts the three PV-FSI sub-indices (HSI, TNI, TEI) and the composite PV-FSI score for every acquisition.
2. Runs a stratified Spearman correlation analysis of TEI against PV-FSI across severity categories (Track A).
3. Fits per-defect-type PCA to derive adaptive sub-index weights and constructs the adaptive composite PV-FSI_A (Track B).
4. Fits per-defect-type irradiance regressions and derives the irradiance-normalized variant PV-FSI_IR (Track C).
5. Produces every figure and table reported in the manuscript at 600 dpi.

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`

```
pip install numpy pandas matplotlib scipy scikit-learn
```

## Usage

1. Obtain the DTU Drone Infrared Thermography Dataset (openly available; see the dataset's own repository for access) and the processed row-level feature file, `dtu_pvfsi_features.csv`.
2. Open `pvfsi_adaptive_study.py` and edit the two paths at the top of the **USER CONFIGURATION** section:
   - `FEATURE_CSV` — path to `dtu_pvfsi_features.csv`
   - `DTU_ROOT` — path to the DTU dataset's root folder (set to `None` to run in feature-only mode, which skips the I–V-curve figures)
3. Run:

```
python pvfsi_adaptive_study.py
```

Outputs are written to `pvfsi_adaptive_outputs/figures/` (PNG, 600 dpi) and `pvfsi_adaptive_outputs/tables/` (CSV).

## Data availability

This code does not redistribute the DTU dataset. See the manuscript's Data Availability statement for the dataset citation and access details.

## Citation

If you use this code, please cite the manuscript (citation details to be added upon publication).

## License

MIT — see `LICENSE`.
