"""
pvfsi_adaptive_study.py
=======================
Full analysis code for:
  "Adaptive PV-FSI: Defect-Specific Sub-Index Weighting and
   Irradiance-Normalised Severity Assessment for UAV-Based
   Photovoltaic Thermographic Inspection"

USER CONFIGURATION  (only section you need to edit)
─────────────────────────────────────────────────────
  FEATURE_CSV  path to dtu_pvfsi_features.csv
  DTU_ROOT     root folder of the DTU dataset containing
               sub-folders 01_Cell Cracks/, 02_Cell Interconnects/,
               …, 07_Soiling 2/
               Set to None to run in FEATURE-ONLY mode
               (all Tracks run; I-V figures skipped)

Outputs
─────────────────────────────────────────────────────
  {OUT_DIR}/figures/  –  fig_01 … fig_09  at 600 dpi, font 11
  {OUT_DIR}/tables/   –  table_01 … table_S1  as .csv
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, sys, re, warnings, logging

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════════════════════
# ① USER CONFIGURATION  ← edit only here
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_CSV = r"C:\Users\masud\Desktop\Thermal PV FSI\PVFSI_THermoelectric_Validation_Datasets\DTU_outputs\dtu_pvfsi_features.csv"
DTU_ROOT    = r"C:\Users\masud\Desktop\Thermal PV FSI\PVFSI_THermoelectric_Validation_Datasets\DTU PV plant - Drone Infrared Thermography Dataset"   # e.g. r"C:\Users\masud\...\DTU PV plant - Drone Infrared Thermography Dataset"
OUT_DIR     = "pvfsi_adaptive_outputs"

# ══════════════════════════════════════════════════════════════════════════════
# ② GLOBAL STYLE
# ══════════════════════════════════════════════════════════════════════════════
FONT_SIZE = 11
DPI       = 600
matplotlib.rcParams.update({
    "font.family":             "Liberation Sans",
    "font.size":                FONT_SIZE,
    "axes.titlesize":           FONT_SIZE,
    "axes.labelsize":           FONT_SIZE,
    "xtick.labelsize":          FONT_SIZE,
    "ytick.labelsize":          FONT_SIZE,
    "legend.fontsize":          FONT_SIZE,
    "figure.titlesize":         FONT_SIZE,
    "figure.max_open_warning":  0,
    "pdf.fonttype":             42,
    "svg.fonttype":             "none",
})

C_BLUE   = "#1F3864";  C_TEAL  = "#1A5276";  C_CORAL  = "#922B21"
C_AMBER  = "#B7770D";  C_GREEN = "#1E8449";  C_PURPLE = "#6C3483"
C_GRAY   = "#717D7E"

CAT_COLOURS = {"Healthy": "#2ECC71", "Mild": "#F39C12",
               "Moderate": "#E74C3C", "Severe": "#7B241C"}
CAT_ORDER   = ["Healthy", "Mild", "Moderate", "Severe"]

DEFECT_SHORT = {
    "01_Cell Cracks":                   "Cell Cracks",
    "02_Cell Interconnects":            "Cell Interconn.",
    "03_PID":                           "PID",
    "04_Glass Cracks":                  "Glass Cracks",
    "05_Short-circuited Bypass Diode":  "Bypass Diode",
    "06_Soiling 1":                     "Soiling 1",
    "07_Soiling 2":                     "Soiling 2",
}
DEFECT_ORDER = list(DEFECT_SHORT.keys())
CRACK_FAMILY = ["01_Cell Cracks","02_Cell Interconnects","03_PID","04_Glass Cracks"]
HI_FAMILY    = ["05_Short-circuited Bypass Diode","06_Soiling 1","07_Soiling 2"]

IV_SUFFIX = {
    "01_Cell Cracks":                   "CellCracks",
    "02_Cell Interconnects":            "Interconnects",
    "03_PID":                           "PID",
    "04_Glass Cracks":                  "GlassCrack",
    "05_Short-circuited Bypass Diode":  "BPD",
}

# ══════════════════════════════════════════════════════════════════════════════
# ③ OUTPUT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    [fig] {name}")

def savetab(df_t, name, index=True):
    path = os.path.join(TAB_DIR, name)
    df_t.to_csv(path, index=index)
    print(f"    [tab] {name}")

def section(title):
    print(f"\n{chr(8212)*70}\n  {title}\n{chr(8212)*70}")

# ══════════════════════════════════════════════════════════════════════════════
# ④ DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
section("0. Loading feature data")

if not os.path.isfile(FEATURE_CSV):
    sys.exit(f"ERROR: FEATURE_CSV not found: {FEATURE_CSV}")

df = pd.read_csv(FEATURE_CSV)

# ── irradiance bin: robust multi-pattern extraction ──────────────────────────
def extract_irr_bin(filename):
    """
    Handles three filename patterns in the DTU dataset:
      T14_1_7_10m_200.raw          → 200
      T10_10m_200_APV739.raw       → 200
      T_200_10m - Copy_PD.raw      → 200  (soiling)
    Also maps abv800 correctly.
    """
    # Pattern 1 & 2: _200.raw  _600.raw  _abv800.raw  _200_APV.raw
    m = re.search(r"[_-](abv\d+|\d{3,4})(?:_[A-Z0-9]+)?\.raw$",
                  filename, re.IGNORECASE)
    if m:
        val = m.group(1)
        if val in ("200","600","800","abv800"):
            return val
    # Pattern 3 soiling: T_200_10m...
    m2 = re.search(r"T_(\d{3,4})_", filename)
    if m2 and m2.group(1) in ("200","600","800"):
        return m2.group(1)
    return None

df["irr_bin"] = df["filename"].apply(extract_irr_bin)
irr_map = {"200": 200, "600": 600, "800": 800, "abv800": 1000}
df["irr_num"] = df["irr_bin"].map(irr_map)

# ── convenience columns ───────────────────────────────────────────────────────
df["T_spread"]     = df["Tmax"] - df["Tmean"]
df["defect_short"] = df["defect"].map(DEFECT_SHORT)
df["defect"]       = pd.Categorical(df["defect"],
                                    categories=DEFECT_ORDER, ordered=True)
df["FSI_cat_ord"]  = pd.Categorical(df["FSI_category"],
                                    categories=CAT_ORDER, ordered=True)
df = df.sort_values(["defect","altitude_m"]).reset_index(drop=True)

SUBIDX = ["HSI_norm","TEI_norm","TNI_norm"]

print(f"  Rows: {len(df)}  |  Defects: {df['defect'].nunique()}")
print(f"  Pmpp available: {df['Pmpp'].notna().sum()} rows")
print(f"  irr_bin coverage: {df['irr_bin'].value_counts().to_dict()}")
print(f"  FSI_category: {df['FSI_category'].value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════════════════════
# ⑤ I-V CURVE LOADING & PARAMETER EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
section("1. I-V curve loading")

IV_AVAILABLE = False

def load_iv_csv(path):
    """
    Load a V-I sweep CSV robustly.
    Handles comma/semicolon delimiters, header rows, V/I column order,
    and Voltage/Current named columns.
    Returns (V_array, I_array) sorted by ascending V, or (None, None).
    """
    if not os.path.isfile(path):
        return None, None
    for sep in [",", ";", "\t"]:
        try:
            tmp = pd.read_csv(path, sep=sep, header=None, comment="#",
                              engine="python")
            tmp = tmp.apply(pd.to_numeric, errors="coerce").dropna()
            if tmp.shape[1] >= 2 and len(tmp) >= 5:
                V, I = tmp.iloc[:,0].values.astype(float), \
                       tmp.iloc[:,1].values.astype(float)
                if V[-1] < V[0]:          # ensure ascending voltage
                    V, I = V[::-1], I[::-1]
                if I.mean() > 5 * V.mean() and V.mean() < 2:
                    V, I = I, V            # swap if columns reversed
                return V, I
        except Exception:
            continue
    try:                                   # named-column fallback
        tmp = pd.read_csv(path)
        tmp.columns = [c.lower().strip() for c in tmp.columns]
        vc = next((c for c in tmp.columns if "volt" in c or c == "v"), None)
        ic = next((c for c in tmp.columns if "curr" in c or c == "i"), None)
        if vc and ic:
            V = tmp[vc].astype(float).values
            I = tmp[ic].astype(float).values
            if V[-1] < V[0]:
                V, I = V[::-1], I[::-1]
            return V, I
    except Exception:
        pass
    return None, None


def fit_iv_params(V, I):
    """
    Extract key I-V parameters from a measured sweep.
    Returns dict or None on failure.
    """
    try:
        Voc = float(np.interp(0.0, -I[::-1], V[::-1]))
        Isc = float(np.interp(0.0,  V,        I))
        P   = V * I
        idx = int(np.argmax(P))
        Vmp, Imp = float(V[idx]), float(I[idx])
        Pmpp_fit = float(Vmp * Imp)
        FF  = Pmpp_fit / (Voc * Isc) if Voc * Isc > 0 else np.nan
        # Rs from slope near Voc; Rsh from slope near Isc
        dV  = np.diff(V); dI = np.diff(I)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(np.abs(dI) > 1e-6, dV / dI, np.nan)
        near_voc = V[:-1] > 0.85 * Voc
        near_isc = V[:-1] < 0.15 * Voc
        Rs_est  = float(np.nanmedian(np.abs(slope[near_voc]))) if near_voc.any() else np.nan
        Rsh_est = float(np.nanmedian(np.abs(slope[near_isc]))) if near_isc.any() else np.nan
        return {"Voc": round(Voc,4), "Isc": round(Isc,4),
                "Vmp": round(Vmp,4), "Imp": round(Imp,4),
                "Pmpp_fit": round(Pmpp_fit,4),
                "FF":       round(FF,4) if not np.isnan(FF) else np.nan,
                "Rs_est":   round(Rs_est,4) if not np.isnan(Rs_est) else np.nan,
                "Rsh_est":  round(Rsh_est,2) if not np.isnan(Rsh_est) else np.nan}
    except Exception:
        return None


iv_records = []

if DTU_ROOT is not None:
    if not os.path.isdir(DTU_ROOT):
        print(f"  WARNING: DTU_ROOT not found — running in FEATURE-ONLY mode.")
    else:
        print(f"  DTU_ROOT: {DTU_ROOT}")
        loaded = failed = 0

        # structured defects
        for defect, suffix in IV_SUFFIX.items():
            for mod in sorted(df[df["defect"]==defect]["module_id"].unique()):
                n = mod.split()[-1]
                path = os.path.join(DTU_ROOT, defect, mod,
                                    f"Mod{n}_IV_{suffix}.csv")
                V, I = load_iv_csv(path)
                if V is not None:
                    p = fit_iv_params(V, I)
                    if p:
                        p["defect"] = defect; p["module_id"] = mod
                        iv_records.append(p); loaded += 1
                    else:
                        failed += 1
                else:
                    failed += 1

        # soiling — filename from iv_csv column
        for defect in ["06_Soiling 1","07_Soiling 2"]:
            rows = df[(df["defect"]==defect) & df["iv_csv"].notna()] \
                     .drop_duplicates("module_id")
            for _, row in rows.iterrows():
                fname = row["iv_csv"].replace("\\","/").split("/")[-1]
                path  = os.path.join(DTU_ROOT, defect, row["module_id"], fname)
                V, I  = load_iv_csv(path)
                if V is not None:
                    p = fit_iv_params(V, I)
                    if p:
                        p["defect"] = defect; p["module_id"] = row["module_id"]
                        iv_records.append(p); loaded += 1
                    else:
                        failed += 1
                else:
                    failed += 1

        print(f"  I-V files: {loaded} loaded, {failed} not found / parse error")
        if loaded > 0:
            IV_AVAILABLE = True
            df_iv = pd.DataFrame(iv_records)
            df_iv["defect"] = pd.Categorical(df_iv["defect"],
                                             categories=DEFECT_ORDER, ordered=True)
            df = df.merge(
                df_iv[["defect","module_id","Voc","Isc","Vmp","Imp",
                        "Pmpp_fit","FF","Rs_est","Rsh_est"]],
                on=["defect","module_id"], how="left")
            print(f"  Merged: FF={df['FF'].notna().sum()} rows, "
                  f"Voc={df['Voc'].notna().sum()} rows")
        else:
            print("  No I-V files loaded — FEATURE-ONLY mode.")
else:
    print("  DTU_ROOT = None → FEATURE-ONLY mode.")
    print("  Assign DTU_ROOT to enable I-V analysis (FF, Voc, Isc, Rs, Rsh).")

# add empty IV columns if not present
for col in ["Voc","Isc","Vmp","Imp","Pmpp_fit","FF","Rs_est","Rsh_est"]:
    if col not in df.columns:
        df[col] = np.nan

# ══════════════════════════════════════════════════════════════════════════════
# ⑥ POWER-LOSS & ELECTRICAL COHERENCE
# ══════════════════════════════════════════════════════════════════════════════
section("2. Power-loss and I-V parameter analysis")

# exclude Glass Cracks (different rated module, ~529 W vs ~290 W)
df_pw = df[df["defect"] != "04_Glass Cracks"].copy()
baseline_pmpp = df_pw[df_pw["FSI_category"]=="Healthy"]["Pmpp"].mean()

# ── 2-A: Pmpp-based power loss per severity tier ─────────────────────────────
rows_pl = []
for cat in CAT_ORDER:
    g = df_pw[df_pw["FSI_category"]==cat]["Pmpp"].dropna()
    if len(g) == 0:
        continue
    loss = (baseline_pmpp - g.mean()) / baseline_pmpp * 100 \
           if not np.isnan(baseline_pmpp) else np.nan
    rows_pl.append({"FSI_category": cat, "n_Pmpp": len(g),
                    "Mean_Pmpp_W": round(g.mean(),2), "Std_Pmpp_W": round(g.std(),2),
                    "Pmpp_loss_pct": round(loss,2) if not np.isnan(loss) else np.nan})
tab_pl = pd.DataFrame(rows_pl).set_index("FSI_category")
savetab(tab_pl, "table_02_power_loss_per_severity.csv")

# ── 2-B: FF-based loss per severity tier (if IV available) ───────────────────
tab_ff = pd.DataFrame()
if IV_AVAILABLE and df["FF"].notna().sum() > 10:
    baseline_ff = df_pw[df_pw["FSI_category"]=="Healthy"]["FF"].mean()
    rows_ff = []
    for cat in CAT_ORDER:
        g = df_pw[df_pw["FSI_category"]==cat]["FF"].dropna()
        if len(g) == 0:
            continue
        drop = (baseline_ff - g.mean()) / baseline_ff * 100 \
               if not np.isnan(baseline_ff) else np.nan
        rows_ff.append({"FSI_category": cat, "n_FF": len(g),
                         "Mean_FF": round(g.mean(),4), "Std_FF": round(g.std(),4),
                         "FF_drop_pct": round(drop,2)})
    tab_ff = pd.DataFrame(rows_ff).set_index("FSI_category")
    savetab(tab_ff, "table_08_ff_loss_per_severity.csv")

# ── 2-C: per-defect I-V parameter summary ────────────────────────────────────
iv_cols_sum = ["Voc","Isc","FF","Rs_est","Rsh_est","Pmpp_fit"]
rows_iv = []
for d in DEFECT_ORDER:
    sub = df[df["defect"]==d]
    row = {"Defect": DEFECT_SHORT[d],
           "n": len(sub),
           "Pmpp_STC_mean": round(sub["Pmpp"].mean(),2) if sub["Pmpp"].notna().any() else np.nan,
           "Pmpp_STC_std":  round(sub["Pmpp"].std(),2)  if sub["Pmpp"].notna().sum()>1 else np.nan}
    for col in iv_cols_sum:
        v = sub[col].dropna()
        row[f"{col}_mean"] = round(v.mean(),4) if len(v)>0 else np.nan
        row[f"{col}_std"]  = round(v.std(),4)  if len(v)>1 else np.nan
    rows_iv.append(row)
tab_iv = pd.DataFrame(rows_iv).set_index("Defect")
savetab(tab_iv, "table_07_iv_parameters_per_defect.csv")

# ── 2-D: PVFSI vs electrical parameters Spearman ─────────────────────────────
rows_ec = []
for col in ["Pmpp","FF","Voc","Isc","Rs_est","Rsh_est"]:
    sub = df[["PVFSI",col]].dropna()
    if len(sub) < 10:
        continue
    rho, p = stats.spearmanr(sub["PVFSI"], sub[col])
    rows_ec.append({"Parameter": col, "n": len(sub),
                    "Spearman_rho": round(rho,4),
                    "p_value": f"{p:.3e}",
                    "Significant_p05": "Yes" if p<0.05 else "No",
                    "Source": "STC scalar" if col=="Pmpp" else
                              ("IV fit" if IV_AVAILABLE else "not available")})
tab_ec = pd.DataFrame(rows_ec).set_index("Parameter")
savetab(tab_ec, "table_02b_pvfsi_electrical_correlations.csv")

if not np.isnan(baseline_pmpp):
    print(f"  Pmpp baseline (excl. Glass Cracks): {baseline_pmpp:.2f} W")
    print(tab_pl.to_string())
else:
    print("  NOTE: No Healthy-category Pmpp found (check data coverage)")
if not tab_ec.empty:
    print("\n  PVFSI–electrical Spearman:")
    print(tab_ec[["n","Spearman_rho","p_value","Significant_p05"]].to_string())

# ══════════════════════════════════════════════════════════════════════════════
# ⑦ TRACK A  –  TEI PARADOX
# ══════════════════════════════════════════════════════════════════════════════
section("3. Track A — TEI diagnostic role")

# global variance & rank
r2g = {c: df[c].corr(df["PVFSI"])**2 for c in SUBIDX}
sg  = {c: stats.spearmanr(df[c], df["PVFSI"]) for c in SUBIDX}
sens_rank = {"HSI_norm":"3rd","TEI_norm":"1st (highest)","TNI_norm":"2nd"}

tab_A1 = pd.DataFrame([{
    "Sub_index": c.replace("_norm",""),
    "Pearson_R2_vs_PVFSI": round(r2g[c],4),
    "Spearman_rho": round(sg[c][0],3),
    "p_value": f"{sg[c][1]:.2e}",
    "Sensitivity_rank_paper": sens_rank[c]} for c in SUBIDX]
).set_index("Sub_index")
savetab(tab_A1, "table_01_subindex_variance_sensitivity.csv")

# sign-reversal by severity
rows_sr = []
for cat in CAT_ORDER:
    sub = df[df["FSI_category"]==cat]
    if len(sub) < 5:
        rows_sr.append({"Severity_category":cat,"n":len(sub),
                         "TEI_PVFSI_rho":np.nan,"p_value":np.nan,
                         "Significant_p05":"—","Role":"Insufficient data"})
        continue
    rho,p = stats.spearmanr(sub["TEI_raw"], sub["PVFSI"])
    rows_sr.append({"Severity_category":cat,"n":len(sub),
                    "TEI_PVFSI_rho":round(rho,3),"p_value":round(p,4),
                    "Significant_p05":"Yes" if p<0.05 else "No",
                    "Role":"Severity driver (+)" if rho>0 else "Severity suppressor (−)"})
tab_A2 = pd.DataFrame(rows_sr).set_index("Severity_category")
savetab(tab_A2, "table_05_tei_pvfsi_sign_reversal.csv")
print("  TEI sign-reversal:")
print(tab_A2[["n","TEI_PVFSI_rho","p_value","Role"]].to_string())

# link TEI inversion to Rsh (bypass diode mechanism) if IV available
if IV_AVAILABLE and df["Rsh_est"].notna().sum() > 5:
    bd = df[df["defect"]=="05_Short-circuited Bypass Diode"][["TEI_raw","Rsh_est"]].dropna()
    if len(bd) > 5:
        rho,p = stats.spearmanr(bd["TEI_raw"], bd["Rsh_est"])
        print(f"\n  Bypass Diode TEI vs Rsh: rho={rho:.3f}, p={p:.4f} "
              f"(mechanistic link, n={len(bd)})")

# ══════════════════════════════════════════════════════════════════════════════
# ⑧ TRACK B  –  DEFECT-ADAPTIVE PCA WEIGHTING
# ══════════════════════════════════════════════════════════════════════════════
section("4. Track B — defect-adaptive weighting")

pca_results = {}
for defect in DEFECT_ORDER:
    sub = df[df["defect"]==defect][SUBIDX].dropna()
    if len(sub) < 10:
        continue
    pca = PCA(n_components=3)
    pca.fit(sub)
    w      = np.abs(pca.components_[0])
    w_norm = w / w.sum()
    pca_results[defect] = {
        "HSI": round(w_norm[0],3), "TEI": round(w_norm[1],3),
        "TNI": round(w_norm[2],3),
        "PC1_var": round(pca.explained_variance_ratio_[0],3),
        "Family": "Crack" if defect in CRACK_FAMILY else "High-intensity"}

pca_glob = PCA(n_components=3)
pca_glob.fit(df[SUBIDX].dropna())
wg = np.abs(pca_glob.components_[0]); wg /= wg.sum()

rows_B = [{
    "Defect": DEFECT_SHORT[d], "Family": v["Family"],
    "HSI_weight": v["HSI"], "TEI_weight": v["TEI"], "TNI_weight": v["TNI"],
    "PC1_var_exp": v["PC1_var"]} for d,v in pca_results.items()]
rows_B += [
    {"Defect":"Global fixed (published)","Family":"—","HSI_weight":0.400,
     "TEI_weight":0.300,"TNI_weight":0.300,"PC1_var_exp":"—"},
    {"Defect":"Global PCA (published)","Family":"—",
     "HSI_weight":round(wg[0],3),"TEI_weight":round(wg[1],3),
     "TNI_weight":round(wg[2],3),
     "PC1_var_exp":round(pca_glob.explained_variance_ratio_[0],3)}]
tab_B = pd.DataFrame(rows_B).set_index("Defect")
savetab(tab_B, "table_03_per_defect_pca_weights.csv")

def adaptive_pvfsi(row):
    d = row["defect"]
    if d in pca_results:
        w = pca_results[d]
        return 100*(w["HSI"]*row["HSI_norm"] + w["TEI"]*row["TEI_norm"]
                   + w["TNI"]*row["TNI_norm"])
    return np.nan

df["PVFSI_adaptive"] = df.apply(adaptive_pvfsi, axis=1)

grp_o = [df[df["defect"]==d]["PVFSI"].dropna().values          for d in pca_results]
grp_a = [df[df["defect"]==d]["PVFSI_adaptive"].dropna().values for d in pca_results]
H_o,p_o = stats.kruskal(*grp_o)
H_a,p_a = stats.kruskal(*grp_a)

cv  = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
rf  = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
sub_rf = df[SUBIDX+["altitude_m","defect","FSI_category",
                     "PVFSI","PVFSI_adaptive"]].dropna()
f1_so = cross_val_score(rf,sub_rf[["PVFSI"]],         sub_rf["FSI_category"],cv=cv,scoring="f1_weighted").mean()
f1_sa = cross_val_score(rf,sub_rf[["PVFSI_adaptive"]],sub_rf["FSI_category"],cv=cv,scoring="f1_weighted").mean()
f1_to = cross_val_score(rf,sub_rf[SUBIDX+["altitude_m"]],sub_rf["defect"],   cv=cv,scoring="f1_weighted").mean()

tab_B4 = pd.DataFrame([
    {"Metric":"KW H (defect separability)","Original_PVFSI":round(H_o,2),"Adaptive_PVFSI":round(H_a,2)},
    {"Metric":"KW p-value","Original_PVFSI":f"{p_o:.2e}","Adaptive_PVFSI":f"{p_a:.2e}"},
    {"Metric":"Severity F1 (RF 10-fold)","Original_PVFSI":round(f1_so,3),"Adaptive_PVFSI":round(f1_sa,3)},
    {"Metric":"Defect-type F1 (RF 10-fold)","Original_PVFSI":round(f1_to,3),"Adaptive_PVFSI":"—"},
]).set_index("Metric")
savetab(tab_B4, "table_04_separability_comparison.csv")
print(f"  KW H: original={H_o:.2f}  adaptive={H_a:.2f}")
print(f"  Severity F1: original={f1_so:.3f}  adaptive={f1_sa:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# ⑨ TRACK C  –  PV-FSI* IRRADIANCE NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════
section("5. Track C — irradiance normalisation")

rows_sl = []
for defect in DEFECT_ORDER:
    sub = df[(df["defect"]==defect) & df["irr_num"].notna()][["irr_num","PVFSI"]].dropna()
    if len(sub) < 8:
        rows_sl.append({"Defect":DEFECT_SHORT[defect],"Slope_per_100Wm2":np.nan,
                         "R2":np.nan,"p_value":np.nan,"Note":"No irr data"}); continue
    sl,ic,r,p,_ = stats.linregress(sub["irr_num"], sub["PVFSI"])
    rows_sl.append({"Defect":DEFECT_SHORT[defect],"Slope_per_100Wm2":round(sl*100,3),
                    "R2":round(r**2,3),"p_value":round(p,4),"Note":""})
tab_C = pd.DataFrame(rows_sl).set_index("Defect")
savetab(tab_C, "table_06_irradiance_slopes.csv")

def compute_pvfsi_star(df_in):
    out = df_in.copy(); out["PVFSI_star"] = np.nan
    for defect in DEFECT_ORDER:
        mask = (out["defect"]==defect) & out["irr_num"].notna()
        sub  = out[mask][["irr_num","PVFSI"]].dropna()
        if len(sub) < 8: continue
        sl,ic,*_ = stats.linregress(sub["irr_num"], sub["PVFSI"])
        out.loc[mask,"PVFSI_star"] = out.loc[mask,"PVFSI"] - \
                                     (sl*out.loc[mask,"irr_num"] + ic)
    return out

df = compute_pvfsi_star(df)

def kw_irr(col):
    grps = [df[(df["irr_num"]==v) & df[col].notna()][col].values
            for v in [200,600,800,1000]]
    return stats.kruskal(*[g for g in grps if len(g)>1])
def kw_sev(col):
    grps = [df[(df["FSI_category"]==c) & df[col].notna()][col].values
            for c in CAT_ORDER]
    return stats.kruskal(*[g for g in grps if len(g)>1])

H_io,p_io = kw_irr("PVFSI");     H_is,p_is = kw_irr("PVFSI_star")
H_so2,p_so2 = kw_sev("PVFSI");  H_ss,p_ss = kw_sev("PVFSI_star")

tab_C4 = pd.DataFrame([
    {"Test":"KW across irr bins","PVFSI":f"H={H_io:.2f}, p={p_io:.4f}","PVFSI_star":f"H={H_is:.2f}, p={p_is:.4f}"},
    {"Test":"KW across severity","PVFSI":f"H={H_so2:.2f}, p={p_so2:.2e}","PVFSI_star":f"H={H_ss:.2f}, p={p_ss:.2e}"},
]).set_index("Test")
savetab(tab_C4, "table_06b_normalisation_kw_tests.csv")
print(f"  KW irr: PVFSI H={H_io:.2f}  →  PVFSI* H={H_is:.2f}")
print(f"  Severity preserved: PVFSI* H={H_ss:.2f}, p={p_ss:.2e}")

# healthy baseline table
irr_baseline = (
    df[(df["FSI_category"]=="Healthy") & df["irr_num"].notna()]
    .groupby("irr_num")["PVFSI"]
    .agg(["mean","std","count"])
    .rename(columns={"mean":"baseline_mean","std":"baseline_std","count":"n"}))
savetab(irr_baseline, "table_06c_healthy_irr_baseline.csv")

# ══════════════════════════════════════════════════════════════════════════════
# ⑩ COMPREHENSIVE DESCRIPTIVE TABLE
# ══════════════════════════════════════════════════════════════════════════════
section("6. Comprehensive descriptive stats")

rows_d = []
for defect in DEFECT_ORDER:
    sub = df[df["defect"]==defect]
    row = {"Defect": DEFECT_SHORT[defect], "n": len(sub),
           "PVFSI_mean": round(sub["PVFSI"].mean(),2),
           "PVFSI_std":  round(sub["PVFSI"].std(),2),
           "TEI_mean": round(sub["TEI_raw"].mean(),3),
           "HSI_mean": round(sub["HSI_raw"].mean(),3),
           "TNI_mean": round(sub["TNI_raw"].mean(),4),
           "Tmax_mean": round(sub["Tmax"].mean(),2),
           "Tstd_mean": round(sub["Tstd"].mean(),3),
           "Dominant_cat": sub["FSI_category"].value_counts().index[0],
           "Pmpp_STC_mean": round(sub["Pmpp"].mean(),2) if sub["Pmpp"].notna().any() else np.nan,
           "FF_mean":  round(sub["FF"].mean(),4) if IV_AVAILABLE and sub["FF"].notna().any() else np.nan,
           "Voc_mean": round(sub["Voc"].mean(),3) if IV_AVAILABLE and sub["Voc"].notna().any() else np.nan,
           "Isc_mean": round(sub["Isc"].mean(),3) if IV_AVAILABLE and sub["Isc"].notna().any() else np.nan,
           "Rs_mean":  round(sub["Rs_est"].mean(),4) if IV_AVAILABLE and sub["Rs_est"].notna().any() else np.nan,
           "Adaptive_HSI_w": pca_results.get(defect,{}).get("HSI","—"),
           "Adaptive_TEI_w": pca_results.get(defect,{}).get("TEI","—"),
           "Adaptive_TNI_w": pca_results.get(defect,{}).get("TNI","—")}
    rows_d.append(row)
tab_desc = pd.DataFrame(rows_d).set_index("Defect")
savetab(tab_desc, "table_S1_comprehensive_descriptive.csv")

# ══════════════════════════════════════════════════════════════════════════════
# ⑪ FIGURES
# ══════════════════════════════════════════════════════════════════════════════
section("7. Generating figures")

radar_defs = [d for d in DEFECT_ORDER if d in pca_results]

# ── Fig 1: Spearman correlation matrix ───────────────────────────────────────
print("  Fig 1 …")
ccols  = SUBIDX + ["PVFSI"]
clabs  = ["HSI$_{norm}$","TEI$_{norm}$","TNI$_{norm}$","PV-FSI"]
smat   = np.array([[stats.spearmanr(df[a],df[b])[0] for b in ccols] for a in ccols])
fig,ax = plt.subplots(figsize=(5,4.2))
im = ax.imshow(smat, cmap="RdYlGn", vmin=-1, vmax=1)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Spearman ρ"); cb.ax.tick_params(labelsize=FONT_SIZE)
ax.set_xticks(range(4)); ax.set_xticklabels(clabs)
ax.set_yticks(range(4)); ax.set_yticklabels(clabs)
for i in range(4):
    for j in range(4):
        c2 = "white" if abs(smat[i,j])>0.65 else "black"
        ax.text(j,i,f"{smat[i,j]:.3f}",ha="center",va="center",
                fontsize=FONT_SIZE,color=c2,fontweight="bold")
ax.set_title("Spearman Correlation: Sub-indices and PV-FSI")
fig.tight_layout(); savefig(fig,"fig_01_subindex_correlation_matrix.png")

# ── Fig 2: TEI sign reversal ─────────────────────────────────────────────────
print("  Fig 2 …")
fig,axes = plt.subplots(1,4,figsize=(12,3.5),sharey=False)
for ax,cat in zip(axes,CAT_ORDER):
    sub = df[df["FSI_category"]==cat]
    col = CAT_COLOURS[cat]
    ax.scatter(sub["TEI_raw"],sub["PVFSI"],color=col,alpha=0.4,s=12,
               linewidths=0,rasterized=True)
    if len(sub) >= 3:
        rho,p = stats.spearmanr(sub["TEI_raw"],sub["PVFSI"])
        m,b   = np.polyfit(sub["TEI_raw"],sub["PVFSI"],1)
        xr    = np.linspace(sub["TEI_raw"].min(),sub["TEI_raw"].max(),100)
        ax.plot(xr,m*xr+b,color=col,lw=1.8,ls="-" if p<0.05 else "--")
        ps = "p<0.001" if p<0.001 else f"p={p:.3f}"
        ax.set_title(f"{cat}\nρ={rho:+.3f}, {ps}",color=C_BLUE)
    else:
        ax.set_title(f"{cat}\n(n={len(sub)}, insufficient)",color=C_GRAY)
    ax.set_xlabel("TEI$_{raw}$")
    ax.set_ylabel("PV-FSI" if cat=="Healthy" else "")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if cat in ("Moderate","Severe"): ax.set_facecolor("#FDFEFE")
fig.suptitle("TEI–PV-FSI Sign Reversal Across Severity Categories (Track A)",
             fontweight="bold",y=1.02)
fig.tight_layout(); savefig(fig,"fig_02_tei_pvfsi_sign_reversal.png")

# ── Fig 3: Radar — adaptive vs fixed weights ─────────────────────────────────
print("  Fig 3 …")
n_r    = len(radar_defs)
angles = np.linspace(0,2*np.pi,n_r,endpoint=False).tolist()+[0]
sc_r   = [C_CORAL,C_TEAL,C_BLUE]
fw_r   = [0.4,0.3,0.3]
fig,axes = plt.subplots(1,3,figsize=(12,4.2),subplot_kw=dict(polar=True))
for ax,sl,fw,sc in zip(axes,["HSI","TEI","TNI"],fw_r,sc_r):
    va = [pca_results[d][sl] for d in radar_defs]+[pca_results[radar_defs[0]][sl]]
    vf = [fw]*(n_r+1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([DEFECT_SHORT[d] for d in radar_defs],size=FONT_SIZE-1)
    ax.set_yticklabels([]); ax.set_ylim(0,1)
    ax.plot(angles,va,color=sc,lw=1.8); ax.fill(angles,va,color=sc,alpha=0.18)
    ax.plot(angles,vf,color=C_GRAY,lw=1.2,ls="--")
    ax.set_title(f"{sl} weight",fontweight="bold",pad=18,color=sc)
    for rv in [0.25,0.5,0.75]:
        ax.text(0,rv,str(rv),ha="center",va="center",
                fontsize=FONT_SIZE-2,color=C_GRAY)
leg = [Line2D([0],[0],color=c,lw=1.8,label=f"{l} adaptive")
       for l,c in zip(["HSI","TEI","TNI"],sc_r)]
leg += [Line2D([0],[0],color=C_GRAY,lw=1.2,ls="--",label="Fixed (0.4/0.3/0.3)")]
fig.legend(handles=leg,loc="lower center",ncol=4,frameon=False,
           bbox_to_anchor=(0.5,-0.06))
fig.suptitle("Per-Defect Adaptive Weights vs. Fixed Weights (Track B)",fontweight="bold")
fig.tight_layout(rect=[0,0.06,1,1]); savefig(fig,"fig_03_adaptive_weights_radar.png")

# ── Fig 4: Original vs adaptive boxplots ─────────────────────────────────────
print("  Fig 4 …")
x = np.arange(len(radar_defs)); wd = 0.32
fig,ax = plt.subplots(figsize=(11,4.5))
bp1 = ax.boxplot([df[df["defect"]==d]["PVFSI"].dropna().values          for d in radar_defs],
                  positions=x-wd/2,widths=wd*0.9,patch_artist=True,
                  medianprops=dict(color="white",lw=1.5),
                  flierprops=dict(marker="o",ms=3,alpha=0.4))
bp2 = ax.boxplot([df[df["defect"]==d]["PVFSI_adaptive"].dropna().values for d in radar_defs],
                  positions=x+wd/2,widths=wd*0.9,patch_artist=True,
                  medianprops=dict(color="white",lw=1.5),
                  flierprops=dict(marker="^",ms=3,alpha=0.4))
for p in bp1["boxes"]: p.set(facecolor=C_BLUE, alpha=0.65)
for p in bp2["boxes"]: p.set(facecolor=C_CORAL,alpha=0.65)
ax.set_xticks(x); ax.set_xticklabels([DEFECT_SHORT[d] for d in radar_defs])
ax.set_ylabel("PV-FSI score"); ax.set_xlabel("Defect type")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.02,0.97,f"KW H (original)={H_o:.1f}\nKW H (adaptive)={H_a:.1f}",
        transform=ax.transAxes,va="top",fontsize=FONT_SIZE-1,color=C_BLUE,
        bbox=dict(boxstyle="round,pad=0.3",facecolor="white",edgecolor=C_BLUE,alpha=0.8))
ax.legend(handles=[mpatches.Patch(color=C_BLUE, alpha=0.65,label="Original PV-FSI"),
                    mpatches.Patch(color=C_CORAL,alpha=0.65,label="Adaptive PV-FSI_A")],
          frameon=False)
ax.set_title("Original vs. Adaptive PV-FSI by Defect Type (Track B)",fontweight="bold")
fig.tight_layout(); savefig(fig,"fig_04_pvfsi_original_vs_adaptive.png")

# ── Fig 5: Irradiance correction ─────────────────────────────────────────────
print("  Fig 5 …")
irr_defs = [d for d in DEFECT_ORDER
            if df[(df["defect"]==d)&df["irr_num"].notna()].shape[0]>=8]
n_id = len(irr_defs); irr_v=[200,600,800,1000]; irr_l=["200","600","800",">800"]
fig,axes = plt.subplots(2,n_id,figsize=(max(10,n_id*2.4),6.5),sharey="row")
if n_id == 1: axes = axes.reshape(2,1)
for ci,defect in enumerate(irr_defs):
    sub = df[(df["defect"]==defect)&df["irr_num"].notna()]
    for ri,(sc,yl,pc) in enumerate([("PVFSI","PV-FSI score",C_BLUE),
                                      ("PVFSI_star","PV-FSI*",C_TEAL)]):
        ax = axes[ri,ci]
        bp = ax.boxplot([sub[sub["irr_num"]==v][sc].dropna().values for v in irr_v],
                         patch_artist=True,widths=0.55,
                         medianprops=dict(color="white",lw=1.5),
                         flierprops=dict(marker="o",ms=2.5,alpha=0.5))
        for p in bp["boxes"]: p.set(facecolor=pc,alpha=0.6)
        ax.set_xticks(range(1,5)); ax.set_xticklabels(irr_l)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if ci==0: ax.set_ylabel(yl)
        if ri==0: ax.set_title(DEFECT_SHORT[defect],fontweight="bold")
        if ri==1: ax.set_xlabel("Irradiance (W m⁻²)")
axes[0,0].annotate("Before\ncorrection",xy=(-0.30,0.5),xycoords="axes fraction",
                    rotation=90,va="center",ha="center",fontweight="bold",color=C_BLUE)
axes[1,0].annotate("After\ncorrection\n(PV-FSI*)",xy=(-0.30,0.5),
                    xycoords="axes fraction",rotation=90,va="center",ha="center",
                    fontweight="bold",color=C_TEAL)
fig.suptitle("Irradiance Drift: Before vs. After Normalisation (Track C)",fontweight="bold")
fig.tight_layout(rect=[0.05,0,1,0.97]); savefig(fig,"fig_05_irradiance_correction.png")

# ── Fig 6: Severity scale + power-loss tiers ─────────────────────────────────
print("  Fig 6 …")
fig,(ax_top,ax_bot) = plt.subplots(2,1,figsize=(10,5.5),
                                    gridspec_kw={"height_ratios":[1.2,2]})
for lo,hi,lbl,col in [(0,20,"Healthy","#2ECC71"),(20,50,"Mild","#F39C12"),
                       (50,75,"Moderate","#E74C3C"),(75,100,"Severe","#7B241C")]:
    ax_top.barh(0,hi-lo,left=lo,height=0.55,color=col,alpha=0.8)
    ax_top.text((lo+hi)/2,0,lbl,ha="center",va="center",
                fontweight="bold",color="white")
ax_top.set_xlim(0,100); ax_top.set_ylim(-0.6,0.6); ax_top.set_xlabel("PV-FSI score")
ax_top.set_yticks([]); ax_top.spines["top"].set_visible(False)
ax_top.spines["left"].set_visible(False); ax_top.spines["right"].set_visible(False)
ax_top.set_title("Severity Scale and Power-Loss Thresholds",fontweight="bold")

cats_pw = list(tab_pl.index); losses=tab_pl["Pmpp_loss_pct"].values
bcols=[CAT_COLOURS[c] for c in cats_pw]; ns=tab_pl["n_Pmpp"].values
bars = ax_bot.bar(cats_pw,losses,color=bcols,alpha=0.8,edgecolor="white",lw=0.8)
for bar,loss,nv in zip(bars,losses,ns):
    if not np.isnan(loss):
        ax_bot.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.4,
                    f"{loss:.1f}%\n(n={nv})",ha="center",va="bottom",fontweight="bold")
if IV_AVAILABLE and not tab_ff.empty:
    ax2 = ax_bot.twinx()
    ax2.plot(list(tab_ff.index),tab_ff["FF_drop_pct"].values,
             "D--",color=C_PURPLE,lw=1.5,ms=6,label="FF drop %")
    ax2.set_ylabel("Fill Factor drop (%)",color=C_PURPLE)
    ax2.tick_params(axis="y",colors=C_PURPLE)
    ax2.legend(loc="upper right",frameon=False)
ax_bot.axhline(0,color="black",lw=0.8)
ax_bot.set_ylabel("Pmpp loss vs. Healthy (%)")
ax_bot.set_xlabel("Severity category")
ax_bot.spines["top"].set_visible(False); ax_bot.spines["right"].set_visible(False)
valid_losses = [l for l in losses if not np.isnan(l)]
if valid_losses:
    ax_bot.set_ylim(min(valid_losses)-5, max(valid_losses)+12)
    if "Mild" in cats_pw and "Moderate" in cats_pw:
        ml = tab_pl.loc["Mild","Pmpp_loss_pct"]; mol = tab_pl.loc["Moderate","Pmpp_loss_pct"]
        if not np.isnan(ml) and not np.isnan(mol):
            ax_bot.annotate(f"Non-linear jump\n+{abs(mol-ml):.1f} pp",
                            xy=(cats_pw.index("Moderate"),mol),
                            xytext=(cats_pw.index("Moderate")+0.55,(ml+mol)/2),
                            fontsize=FONT_SIZE-1,color=C_CORAL,
                            arrowprops=dict(arrowstyle="->",color=C_CORAL,lw=1.2))
fig.tight_layout(pad=1.5); savefig(fig,"fig_06_severity_scale_power_loss.png")

# ── Fig 7: I-V parameter profiles  (IV mode only) ────────────────────────────
if IV_AVAILABLE and df["FF"].notna().sum() > 5:
    print("  Fig 7 …")
    iv_pc = ["Voc","Isc","FF","Rs_est"]
    iv_yl = ["V$_{oc}$ (V)","I$_{sc}$ (A)","Fill Factor","R$_s$ (Ω)"]
    fig,axes = plt.subplots(2,2,figsize=(10,6))
    for ax,col,yl in zip(axes.flat,iv_pc,iv_yl):
        data   = [df[df["defect"]==d][col].dropna().values for d in DEFECT_ORDER]
        labels = [DEFECT_SHORT[d] for d,v in zip(DEFECT_ORDER,data) if len(v)>0]
        data   = [v for v in data if len(v)>0]
        if not data: ax.set_visible(False); continue
        bp = ax.boxplot(data,patch_artist=True,
                         medianprops=dict(color="white",lw=1.5),
                         flierprops=dict(marker="o",ms=3,alpha=0.4))
        cols_f = [C_TEAL if d in CRACK_FAMILY else C_CORAL
                  for d in DEFECT_ORDER if len(df[df["defect"]==d][col].dropna())>0]
        for p,c in zip(bp["boxes"],cols_f): p.set(facecolor=c,alpha=0.65)
        ax.set_xticks(range(1,len(labels)+1))
        ax.set_xticklabels(labels,rotation=25,ha="right",fontsize=FONT_SIZE-1)
        ax.set_ylabel(yl)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_title(f"{yl.split('(')[0].strip()} by Defect")
    fig.suptitle("I-V Derived Parameters by Defect Type",fontweight="bold")
    fig.legend(handles=[mpatches.Patch(color=C_TEAL,alpha=0.65,label="Crack family"),
                         mpatches.Patch(color=C_CORAL,alpha=0.65,label="High-intensity family")],
               loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(0.5,-0.02))
    fig.tight_layout(rect=[0,0.04,1,1]); savefig(fig,"fig_07_iv_parameters_by_defect.png")

# ── Fig 8: FF vs PVFSI scatter  (IV mode only) ───────────────────────────────
if IV_AVAILABLE and df["FF"].notna().sum() > 10:
    print("  Fig 8 …")
    fams = {"Crack family": CRACK_FAMILY, "High-intensity family": HI_FAMILY}
    fig,axes = plt.subplots(1,2,figsize=(10,4.2))
    for ax,(fn,fd) in zip(axes,fams.items()):
        sub = df[df["defect"].isin(fd)][["PVFSI","FF","defect"]].dropna()
        if len(sub)<5:
            ax.text(0.5,0.5,"Insufficient data",transform=ax.transAxes,ha="center")
            continue
        for d in fd:
            ds = sub[sub["defect"]==d]
            if len(ds)>1:
                ax.scatter(ds["PVFSI"],ds["FF"],alpha=0.5,s=18,label=DEFECT_SHORT[d],rasterized=True)
        m,b = np.polyfit(sub["PVFSI"],sub["FF"],1)
        xr  = np.linspace(sub["PVFSI"].min(),sub["PVFSI"].max(),100)
        fc  = C_TEAL if "Crack" in fn else C_CORAL
        ax.plot(xr,m*xr+b,color=fc,lw=2)
        rho,p = stats.spearmanr(sub["PVFSI"],sub["FF"])
        ps = "p<0.001" if p<0.001 else f"p={p:.3f}"
        ax.set_title(f"{fn}\nρ={rho:+.3f}, {ps}",color=C_BLUE)
        ax.set_xlabel("PV-FSI score"); ax.set_ylabel("Fill Factor")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(fontsize=FONT_SIZE-2,frameon=False)
    fig.suptitle("Fill Factor vs. PV-FSI Thermo-Electrical Coherence by Family",
                 fontweight="bold")
    fig.tight_layout(); savefig(fig,"fig_08_ff_vs_pvfsi.png")

# ── Fig 9: Two-family weight summary bar chart ───────────────────────────────
print("  Fig 9 …")
def mean_w(dlist):
    ws = [pca_results[d] for d in dlist if d in pca_results]
    return (np.mean([w["HSI"] for w in ws]),
            np.mean([w["TEI"] for w in ws]),
            np.mean([w["TNI"] for w in ws])) if ws else (1/3,1/3,1/3)

cw = mean_w(CRACK_FAMILY); hw = mean_w(HI_FAMILY)
cats4  = ["Crack family\n(avg)","High-intensity\nfamily (avg)",
           "Global fixed\n(published)","Global PCA\n(published)"]
allw   = [cw, hw, (0.4,0.3,0.3),
           (round(wg[0],3),round(wg[1],3),round(wg[2],3))]
fig,ax = plt.subplots(figsize=(8,4))
x4 = np.arange(4); w4 = 0.22
for si,(sl,bc) in enumerate(zip(["HSI","TEI","TNI"],[C_CORAL,C_TEAL,C_BLUE])):
    vals = [aw[si] for aw in allw]
    bars = ax.bar(x4+(si-1)*w4,vals,w4,label=sl,color=bc,alpha=0.75)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                f"{v:.2f}",ha="center",va="bottom",fontsize=FONT_SIZE-1)
ax.axhline(1/3,color=C_GRAY,lw=0.8,ls=":",alpha=0.6,label="Equal (0.33)")
ax.set_xticks(x4); ax.set_xticklabels(cats4)
ax.set_ylabel("Sub-index weight"); ax.set_ylim(0,1.02)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(frameon=False,ncol=4,loc="upper right")
ax.set_title("Two-Family Adaptive Weight Summary (Track B)",fontweight="bold")
fig.tight_layout(); savefig(fig,"fig_09_two_family_weight_summary.png")

# ══════════════════════════════════════════════════════════════════════════════
# ⑫ FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section("8. Study complete")
print(f"\n  Mode: {'FULL (features + I-V curves)' if IV_AVAILABLE else 'FEATURE-ONLY (set DTU_ROOT to enable I-V analysis)'}")
print(f"\n  Figures → {FIG_DIR}/")
for f in sorted(os.listdir(FIG_DIR)): print(f"    {f}")
print(f"\n  Tables  → {TAB_DIR}/")
for f in sorted(os.listdir(TAB_DIR)): print(f"    {f}")

# tidy numerical summary
pl_mild = tab_pl.loc["Mild","Pmpp_loss_pct"] if "Mild" in tab_pl.index else np.nan
pl_mod  = tab_pl.loc["Moderate","Pmpp_loss_pct"] if "Moderate" in tab_pl.index else np.nan
pl_sev  = tab_pl.loc["Severe","Pmpp_loss_pct"] if "Severe" in tab_pl.index else np.nan

print(f"""
  Key results
  {'─'*60}
  Track A
    TEI R² vs PV-FSI  : {r2g['TEI_norm']:.4f}
    TNI R² vs PV-FSI  : {r2g['TNI_norm']:.4f}
    Sign reversal     : Healthy→Moderate  (see table_05)

  Track B
    KW H original     : {H_o:.2f}
    KW H adaptive     : {H_a:.2f}
    Severity F1       : original={f1_so:.3f}  adaptive={f1_sa:.3f}

  Track C
    KW irr PVFSI      : H={H_io:.2f}, p={p_io:.4f}
    KW irr PVFSI*     : H={H_is:.2f}, p={p_is:.4f}
    KW sev PVFSI*     : H={H_ss:.2f}, p={p_ss:.2e}

  Power loss (Pmpp-based, excl. Glass Cracks)
    Mild     : {pl_mild:.1f}% loss
    Moderate : {pl_mod:.1f}% loss
    Severe   : {pl_sev:.1f}% loss
    Non-linear jump Mild→Moderate : {abs(pl_mod-pl_mild):.1f} pp
""")
