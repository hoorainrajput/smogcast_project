# Hourly Smog Nowcasting v3 - research-grade training & evaluation.
#
# What this script does (all numbers in the README come from here):
#   1. Rich leakage-free features (features.py).
#   2. Rolling-origin time-series CV on 5 held-out months, INCLUDING three
#      winter smog months (Nov 2025, Dec 2025, Jan 2026), a monsoon month
#      and the original Oct 2026 test month.  A 24h purge gap separates
#      train from test.
#   3. Four baselines: persistence, seasonal-naive, 24h mean, climatology.
#   4. Two model modes: A = past-only nowcast, B = + forecast weather.
#   5. Ablation study (cumulative feature groups).
#   6. 80% prediction intervals: raw quantile models vs conformalised (CQR).
#   7. Smog-episode classifier (PM2.5 > 55.4 at t+h).
#   8. Noise-sensitivity of forecast-weather inputs.
#   9. Day-block bootstrap CIs for model-vs-baseline differences.
#  10. Final models + metadata for the Streamlit app.
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

from features import (CITIES, FX_COLS, G_LAG, G_PM10, G_ROLL, G_WX,
                      PAST_COLS, UNHEALTHY, build_past_features,
                      future_weather)

FAST = os.environ.get("FAST") == "1"      # smoke-test mode
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "pakistan_air_quality_hourly.csv")
MODEL_DIR = os.path.join(BASE, "models_v3")
CHART_DIR = os.path.join(BASE, "charts_v3")
for p in (MODEL_DIR, CHART_DIR):
    os.makedirs(p, exist_ok=True)

HORIZONS = [6, 24] if FAST else [1, 3, 6, 9, 12, 18, 24]
CQR_H = [6, 24]
CLF_H = [6, 12, 24] if not FAST else [6, 24]
MAX_ITER = 40 if FAST else 250
ALL_FOLDS = {
    "Nov 2025": ("2025-11-01", "2025-12-01"),
    "Dec 2025": ("2025-12-01", "2026-01-01"),
    "Jan 2026": ("2026-01-01", "2026-02-01"),
    "Jul 2026": ("2026-07-01", "2026-08-01"),
    "Oct 2026": ("2026-10-01", "2026-11-02"),
}
FOLDS = dict(list(ALL_FOLDS.items())[:1] + list(ALL_FOLDS.items())[-1:]) \
    if FAST else ALL_FOLDS
GAP = pd.Timedelta(hours=24)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
COL = {"persist": "#9aa0a6", "snaive": "#c58af9", "mean24": "#f6ae2d",
       "clim": "#6d6875", "A": "#2a9d8f", "B": "#e63946"}

# ------------------------------------------------------------------ data
raw = pd.read_csv(DATA, parse_dates=["time"])
full = build_past_features(raw)
d = full.dropna(subset=PAST_COLS).reset_index(drop=True)
log(f"rows {len(raw)}, usable {len(d)}, features {len(PAST_COLS)} past + "
    f"{len(FX_COLS)} forecast-weather")
T = d["time"]

Y, FX, YLOG = {}, {}, {}
for h in HORIZONS:
    Y[h] = d.groupby("city")["pm2_5"].shift(-h)
    FX[h] = future_weather(d, h)
    YLOG[h] = np.log1p(Y[h])
SNAIVE = {h: d.groupby("city")["pm2_5"].shift(24 - h) for h in HORIZONS}


def frame(h, mode, cols=None):
    cols = cols if cols is not None else PAST_COLS
    X = d[cols]
    if mode == "B":
        X = pd.concat([X, FX[h]], axis=1)
    valid = Y[h].notna() & FX[h].notna().all(axis=1)
    return X, valid


def reg():
    return HistGradientBoostingRegressor(
        max_iter=MAX_ITER, learning_rate=0.06, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=False, random_state=0)


def qreg(q):
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=MAX_ITER, learning_rate=0.06,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
        random_state=0)


def clf():
    return HistGradientBoostingClassifier(
        max_iter=MAX_ITER, learning_rate=0.06, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=False, random_state=0)


def fold_masks(start, end):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return (T < s - GAP), ((T >= s) & (T < e))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def predict(m, X):
    return np.clip(np.expm1(m.predict(X)), 0, None)


# ------- forecast-weather noise (used for robust training & evaluation)
NT = 0.4                       # training/eval noise: fraction of each std
NOISE = [0.0, 0.25, NT, 0.5, 1.0]
STD = {c: float(d[c.replace("fx_", "", 1)].std()) for c in FX_COLS
       if c != "fx_wind_rm6"}
STD["fx_wind_rm6"] = float(d["wind_speed_10m"].std())


def perturb(X, k, seed=1):
    X = X.copy()
    r = np.random.default_rng(seed)
    for c in FX_COLS:
        X[c] = X[c] + r.normal(0, k * STD[c], len(X))
    X["fx_relative_humidity_2m"] = X["fx_relative_humidity_2m"].clip(0, 100)
    for c in ("fx_precipitation", "fx_wind_speed_10m", "fx_wind_rm6"):
        X[c] = X[c].clip(0, None)
    return X


# ------------------------------------------------ 1. main CV (A, B, baselines)
rows, models_B_h24 = [], {}
for fname, (s, e) in FOLDS.items():
    tr_m, te_m = fold_masks(s, e)
    for h in HORIZONS:
        XA, valid = frame(h, "A")
        XB, _ = frame(h, "B")
        tr, te = tr_m & valid, te_m & valid
        mA = reg().fit(XA[tr], YLOG[h][tr])
        mB = reg().fit(XB[tr], YLOG[h][tr])
        if h == 24:
            models_B_h24[fname] = mB
        # climatology from the training window only
        clim = d.loc[tr_m].groupby(["city", "month", "hour"])["pm2_5"].mean()
        tt = d.loc[te, "time"] + pd.Timedelta(hours=h)
        key = pd.MultiIndex.from_arrays(
            [d.loc[te, "city"], tt.dt.month, tt.dt.hour])
        out = pd.DataFrame({
            "fold": fname, "h": h, "city": d.loc[te, "city"],
            "time": d.loc[te, "time"], "actual": Y[h][te],
            "persist": d.loc[te, "pm2_5"], "snaive": SNAIVE[h][te],
            "mean24": d.loc[te, "rm24"], "clim": clim.reindex(key).values,
            "predA": predict(mA, XA[te]), "predB": predict(mB, XB[te])})
        rows.append(out)
    log(f"CV fold {fname} done")
cv = pd.concat(rows, ignore_index=True)
cv["clim"] = cv["clim"].fillna(cv["mean24"])
METHODS = ["persist", "snaive", "mean24", "clim", "predA", "predB"]

metrics = {"horizons": HORIZONS, "folds": list(FOLDS)}
pooled = {}
for h in HORIZONS:
    sub = cv[cv["h"] == h]
    pooled[h] = {m: {"MAE": round(mae(sub["actual"], sub[m]), 2),
                     "RMSE": round(rmse(sub["actual"], sub[m]), 2)}
                 for m in METHODS}
    best_base = min(pooled[h][m]["MAE"] for m in METHODS[:4])
    pooled[h]["best_baseline_MAE"] = best_base
    pooled[h]["skill_B_vs_best_baseline_%"] = round(
        100 * (1 - pooled[h]["predB"]["MAE"] / best_base), 1)
    pooled[h]["skill_B_vs_persistence_%"] = round(
        100 * (1 - pooled[h]["predB"]["MAE"] / pooled[h]["persist"]["MAE"]), 1)
metrics["pooled_cv"] = pooled

per_fold = {}
for fname in FOLDS:
    per_fold[fname] = {}
    for h in HORIZONS:
        sub = cv[(cv["fold"] == fname) & (cv["h"] == h)]
        per_fold[fname][h] = {m: round(mae(sub["actual"], sub[m]), 2)
                              for m in METHODS}
metrics["per_fold_MAE"] = per_fold

by_city = {}
for h in (6, 24):
    by_city[h] = {}
    for c in CITIES:
        sub = cv[(cv["h"] == h) & (cv["city"] == c)]
        by_city[h][c] = {m: round(mae(sub["actual"], sub[m]), 2)
                         for m in ("persist", "predA", "predB")}
metrics["by_city_MAE"] = by_city
log("pooled h24:", json.dumps(pooled[24]))

# ---- day-block bootstrap (are the improvements real?) ----
rng = np.random.default_rng(0)
boot = {}
for h in HORIZONS:
    sub = cv[cv["h"] == h].copy()
    sub["day"] = sub["time"].dt.normalize().astype(str) + "|" + sub["fold"]
    for name, a, b in (("B_vs_persist", "persist", "predB"),
                       ("B_vs_A", "predA", "predB")):
        diff = (sub["actual"] - sub[a]).abs() - (sub["actual"] - sub[b]).abs()
        grp = diff.groupby(sub["day"])
        S, N = grp.sum().values, grp.count().values
        idx = rng.integers(0, len(S), size=(2000, len(S)))
        est = S[idx].sum(1) / N[idx].sum(1)
        boot[f"h{h}_{name}"] = {
            "mean_MAE_gain": round(float(S.sum() / N.sum()), 2),
            "ci95": [round(float(np.percentile(est, 2.5)), 2),
                     round(float(np.percentile(est, 97.5)), 2)]}
metrics["bootstrap_MAE_gain"] = boot
log("bootstrap h24:", boot["h24_B_vs_persist"])

# ---------------------------------------------------------- 2. ablation
ABL = [("Lags + hour/month + city", G_LAG),
       ("+ rolling stats & trends", G_LAG + G_ROLL),
       ("+ PM10", G_LAG + G_ROLL + G_PM10)]
ablation = {}
for h in (6, 24):
    ablation[h] = {}
    for label, cols in ABL:
        preds, acts = [], []
        for fname, (s, e) in FOLDS.items():
            tr_m, te_m = fold_masks(s, e)
            X, valid = frame(h, "A", cols)
            tr, te = tr_m & valid, te_m & valid
            m = reg().fit(X[tr], YLOG[h][tr])
            preds.append(predict(m, X[te]))
            acts.append(Y[h][te].values)
        ablation[h][label] = round(mae(np.concatenate(acts),
                                       np.concatenate(preds)), 2)
    sub = cv[cv["h"] == h]
    ablation[h]["+ current weather (= mode A)"] = pooled[h]["predA"]["MAE"]
    ablation[h]["+ forecast weather (= mode B)"] = pooled[h]["predB"]["MAE"]
    ablation[h]["Persistence baseline"] = pooled[h]["persist"]["MAE"]
metrics["ablation_MAE"] = ablation
log("ablation:", json.dumps(ablation))

# ------------------------------------- 3. prediction intervals (80%, CQR)
ALPHA = 0.2


def conformal_q(scores, alpha=ALPHA):
    n = len(scores)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(scores)[k - 1])


def interval_models(h, fit_mask, X):
    lo = qreg(ALPHA / 2).fit(X[fit_mask], YLOG[h][fit_mask])
    hi = qreg(1 - ALPHA / 2).fit(X[fit_mask], YLOG[h][fit_mask])
    return lo, hi


iv_rows = []
cov = {}
for h in CQR_H:
    cov[h] = {}
    X, valid = frame(h, "B")
    for fname, (s, e) in FOLDS.items():
        tr_m, te_m = fold_masks(s, e)
        cal_start = pd.Timestamp(s) - pd.Timedelta(days=90)
        fit = tr_m & valid & (T < cal_start - GAP)
        cal = tr_m & valid & (T >= cal_start)
        te = te_m & valid
        lo, hi = interval_models(h, fit, X)
        lo_c, hi_c = lo.predict(X[cal]), hi.predict(X[cal])
        yc = YLOG[h][cal].values
        qhat = conformal_q(np.maximum(lo_c - yc, yc - hi_c))
        lo_t, hi_t = lo.predict(X[te]), hi.predict(X[te])
        yt = Y[h][te].values
        raw_lo, raw_hi = np.expm1(lo_t), np.expm1(hi_t)
        cqr_lo = np.clip(np.expm1(lo_t - qhat), 0, None)
        cqr_hi = np.expm1(hi_t + qhat)
        # robust recipe: everything trained/calibrated/tested on noisy fx
        Xf = perturb(X[fit], NT, seed=11)
        lo_r, hi_r = qreg(ALPHA / 2).fit(Xf, YLOG[h][fit]), qreg(1 - ALPHA / 2).fit(Xf, YLOG[h][fit])
        mR = reg().fit(Xf, YLOG[h][fit])
        Xc, Xt = perturb(X[cal], NT, seed=12), perturb(X[te], NT, seed=13)
        ycl = YLOG[h][cal].values
        qhat_r = conformal_q(np.maximum(lo_r.predict(Xc) - ycl, ycl - hi_r.predict(Xc)))
        rl = np.clip(np.expm1(lo_r.predict(Xt) - qhat_r), 0, None)
        rh = np.expm1(hi_r.predict(Xt) + qhat_r)
        pbr = predict(mR, Xt)
        cov[h][fname] = {
            "robust_coverage": round(float(np.mean((yt >= rl) & (yt <= rh))), 3),
            "robust_width": round(float(np.mean(rh - rl)), 1),
            "robust_MAE": round(mae(yt, pbr), 2),
            "raw_coverage": round(float(np.mean((yt >= raw_lo) & (yt <= raw_hi))), 3),
            "cqr_coverage": round(float(np.mean((yt >= cqr_lo) & (yt <= cqr_hi))), 3),
            "raw_width": round(float(np.mean(raw_hi - raw_lo)), 1),
            "cqr_width": round(float(np.mean(cqr_hi - cqr_lo)), 1),
            "qhat_log": round(qhat, 3)}
        iv_rows.append(pd.DataFrame({
            "fold": fname, "h": h, "city": d.loc[te, "city"].values,
            "time": d.loc[te, "time"].values, "lo": cqr_lo, "hi": cqr_hi,
            "raw_lo": raw_lo, "raw_hi": raw_hi,
            "lo_r": rl, "hi_r": rh, "predBr": pbr}))
    log(f"intervals h={h} done")
metrics["interval_coverage_80"] = cov
iv = pd.concat(iv_rows, ignore_index=True)
cv_iv = cv.merge(iv, on=["fold", "h", "city", "time"], how="left")

# ------------------------------------------- 4. smog-episode classifier
clf_res, clf_curves, oof_p = {}, {}, []
for h in CLF_H:
    X, valid = frame(h, "B")
    lab = (Y[h] > UNHEALTHY).astype(int)
    ys, ps, psr, per = [], [], [], []
    for fname, (s, e) in FOLDS.items():
        tr_m, te_m = fold_masks(s, e)
        tr, te = tr_m & valid, te_m & valid
        m = clf().fit(X[tr], lab[tr])
        p = m.predict_proba(X[te])[:, 1]
        mr = clf().fit(perturb(X[tr], NT, seed=21), lab[tr])
        pr_ = mr.predict_proba(perturb(X[te], NT, seed=22))[:, 1]
        ys.append(lab[te].values)
        ps.append(p)
        psr.append(pr_)
        oof_p.append(pd.DataFrame({"fold": fname, "h": h,
                                   "city": d.loc[te, "city"].values,
                                   "time": d.loc[te, "time"].values,
                                   "p_exceed_clean": p, "p_exceed": pr_}))
    y_all, p_all = np.concatenate(ys), np.concatenate(ps)
    p_rob = np.concatenate(psr)
    pers_rule = (np.concatenate([d.loc[(fold_masks(*FOLDS[f])[1]) & frame(h, "B")[1],
                                       "pm2_5"].values for f in FOLDS])
                 > UNHEALTHY).astype(int)
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_all, (p_all >= g).astype(int)) for g in grid]
    thr = float(grid[int(np.argmax(f1s))])
    pred = (p_all >= thr).astype(int)
    clf_res[h] = {
        "episode_rate_%": round(100 * float(y_all.mean()), 1),
        "ROC_AUC": round(float(roc_auc_score(y_all, p_all)), 3),
        "avg_precision": round(float(average_precision_score(y_all, p_all)), 3),
        "best_F1_threshold": round(thr, 2),
        "model_precision": round(float(precision_score(y_all, pred)), 3),
        "model_recall": round(float(recall_score(y_all, pred)), 3),
        "model_F1": round(float(f1_score(y_all, pred)), 3),
        "persistence_precision": round(float(precision_score(y_all, pers_rule)), 3),
        "persistence_recall": round(float(recall_score(y_all, pers_rule)), 3),
        "persistence_F1": round(float(f1_score(y_all, pers_rule)), 3)}
    f1r = [f1_score(y_all, (p_rob >= g).astype(int)) for g in grid]
    thr_r = float(grid[int(np.argmax(f1r))])
    pred_r = (p_rob >= thr_r).astype(int)
    clf_res[h].update({
        "robust_ROC_AUC": round(float(roc_auc_score(y_all, p_rob)), 3),
        "robust_threshold": round(thr_r, 2),
        "robust_precision": round(float(precision_score(y_all, pred_r)), 3),
        "robust_recall": round(float(recall_score(y_all, pred_r)), 3),
        "robust_F1": round(float(f1_score(y_all, pred_r)), 3)})
    fpr, tpr, _ = roc_curve(y_all, p_rob)
    clf_curves[h] = (fpr, tpr, float(precision_score(y_all, pers_rule)),
                     float(recall_score(y_all, pers_rule)), y_all, pers_rule)
    log(f"classifier h={h}:", json.dumps(clf_res[h]))
metrics["episode_classifier"] = clf_res
oof_p = pd.concat(oof_p, ignore_index=True)

# --------------------------------------- 5. noise sensitivity (h = 24)
noise = {"clean_trained": {}, "noise_trained": {}}
XB, valid = frame(24, "B")
acc = {(w, k): ([], []) for w in noise for k in NOISE}
for fname, (s, e) in FOLDS.items():
    tr_m, te_m = fold_masks(s, e)
    tr, te = tr_m & valid, te_m & valid
    m_clean = models_B_h24[fname]
    m_noisy = reg().fit(perturb(XB[tr], NT, seed=7), YLOG[24][tr])
    for k in NOISE:
        Xt = perturb(XB[te], k) if k > 0 else XB[te]
        for w, m in (("clean_trained", m_clean), ("noise_trained", m_noisy)):
            acc[(w, k)][0].append(Y[24][te].values)
            acc[(w, k)][1].append(predict(m, Xt))
for (w, k), (a, p) in acc.items():
    noise[w][str(k)] = round(mae(np.concatenate(a), np.concatenate(p)), 2)
metrics["noise_sensitivity_h24_MAE"] = noise
log("noise:", json.dumps(noise))

# ------------------------------- 6. permutation importance (h24, Oct fold)
last = list(FOLDS)[-1]
s, e = FOLDS[last]
tr_m, te_m = fold_masks(s, e)
te = te_m & valid
Xte, yte = XB[te].reset_index(drop=True), Y[24][te].values
base_mae = mae(yte, predict(models_B_h24[last], Xte))
imp = {}
r = np.random.default_rng(3)
for c in Xte.columns:
    gains = []
    for _ in range(3):
        Xp = Xte.copy()
        Xp[c] = r.permutation(Xp[c].values)
        gains.append(mae(yte, predict(models_B_h24[last], Xp)) - base_mae)
    imp[c] = float(np.mean(gains))
imp_s = pd.Series(imp).sort_values(ascending=False)
metrics["permutation_importance_h24"] = {
    k: round(v, 3) for k, v in imp_s.head(15).items()}

# ------------------------------------------------------------- 7. charts
H = HORIZONS
names = {"persist": "Persistence", "snaive": "Seasonal-naive (same hour yesterday)",
         "mean24": "24h mean", "clim": "Climatology", "predA": "Model A (past only)",
         "predB": "Model B (+ forecast weather)"}
key = {"persist": "persist", "snaive": "snaive", "mean24": "mean24",
       "clim": "clim", "predA": "A", "predB": "B"}
fig, ax = plt.subplots(figsize=(9, 4.8))
for m in METHODS:
    ax.plot(H, [pooled[h][m]["MAE"] for h in H], marker="o",
            lw=2.6 if m.startswith("pred") else 1.4,
            color=COL[key[m]], label=names[m])
ax.set_xlabel("Forecast horizon (hours ahead)")
ax.set_ylabel("MAE (µg/m³), pooled over 5 CV folds")
ax.set_title("Error vs horizon: model vs four baselines")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "mae_vs_horizon.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9.5, 4.6))
fn = list(FOLDS)
w = 0.2
for i, m in enumerate(["persist", "snaive", "predA", "predB"]):
    ax.bar(np.arange(len(fn)) + (i - 1.5) * w,
           [per_fold[f][24][m] for f in fn], w, color=COL[key[m]],
           label=names[m])
ax.set_xticks(np.arange(len(fn)))
ax.set_xticklabels(fn)
ax.set_ylabel("MAE at 24h ahead (µg/m³)")
ax.set_title("24h-ahead error per held-out month (winter smog months are the hard test)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "cv_folds_h24.png"), dpi=150)
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, h in zip(axs, (6, 24)):
    labs = list(ablation[h])
    vals = [ablation[h][k] for k in labs]
    colors = [COL["persist"] if k.startswith("Pers") else COL["A"] for k in labs]
    ax.barh(labs[::-1], vals[::-1], color=colors[::-1])
    for y_, v in enumerate(vals[::-1]):
        ax.text(v, y_, f" {v:.2f}", va="center", fontsize=8)
    ax.set_title(f"Ablation: {h}h ahead (MAE)")
    ax.set_xlim(0, max(vals) * 1.15)
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "ablation.png"), dpi=150)
plt.close(fig)

# interval example: worst Lahore week at 24h
ex = cv_iv[(cv_iv["h"] == 24) & (cv_iv["city"] == "Lahore")].dropna(subset=["lo_r"])
ex = ex.sort_values("time")
wk = ex.set_index("time")["actual"].rolling("7D").mean().idxmax()
ex = ex[(ex["time"] > wk - pd.Timedelta(days=7)) & (ex["time"] <= wk)]
tt = ex["time"] + pd.Timedelta(hours=24)
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.fill_between(tt, ex["lo_r"], ex["hi_r"], color=COL["B"], alpha=0.18,
                label="80% prediction interval (conformalised)")
ax.plot(tt, ex["actual"], color="#264653", lw=1.8, label="Actual PM2.5")
ax.plot(tt, ex["predBr"], color=COL["B"], lw=1.8, label="24h-ahead forecast (noisy weather inputs)")
ax.axhline(UNHEALTHY, color="#b00020", ls="--", lw=1, label="Unhealthy threshold (55.4)")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title(f"Lahore, smog week ending {wk:%d %b %Y}: forecast with uncertainty")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "interval_example.png"), dpi=150)
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, h in zip(axs, CQR_H):
    xs = np.arange(len(fn))
    ax.bar(xs - 0.27, [cov[h][f]["raw_coverage"] for f in fn], 0.27,
           color="#bbbbbb", label="Raw quantile models")
    ax.bar(xs, [cov[h][f]["cqr_coverage"] for f in fn], 0.27,
           color=COL["A"], label="Conformalised (clean weather)")
    ax.bar(xs + 0.27, [cov[h][f]["robust_coverage"] for f in fn], 0.27,
           color=COL["B"], label="Conformalised, noisy weather (deployed)")
    ax.axhline(0.8, color="k", ls="--", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(fn, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{h}h ahead: 80% interval coverage")
axs[0].legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "interval_coverage.png"), dpi=150)
plt.close(fig)

top = imp_s.head(12)[::-1]
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.barh(top.index, top.values, color=COL["B"])
ax.set_xlabel("MAE increase when feature is shuffled (µg/m³)")
ax.set_title("Permutation importance, 24h model, Oct 2026 fold")
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "importance.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.8, 5))
for h, c in zip(CLF_H, ["#2a9d8f", "#f6ae2d", "#e63946"]):
    fpr, tpr, pp, pr, y_all, pers_rule = clf_curves[h]
    ax.plot(fpr, tpr, color=c, lw=2, label=f"{h}h ahead (AUC {clf_res[h]['robust_ROC_AUC']})")
    fp = float(((pers_rule == 1) & (y_all == 0)).sum() / max((y_all == 0).sum(), 1))
    ax.scatter([fp], [pr], color=c, marker="x", s=60)
ax.plot([0, 1], [0, 1], color="#bbb", ls="--")
ax.set_xlabel("False-positive rate")
ax.set_ylabel("True-positive rate (recall)")
ax.set_title("Smog-episode detection, noisy forecast weather\n(x = persistence rule)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "episode_roc.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ks = [str(k) for k in NOISE]

ax.plot(NOISE, [noise["clean_trained"][k] for k in ks], marker="o",
        color=COL["B"], label="B trained on clean weather")
ax.plot(NOISE, [noise["noise_trained"][k] for k in ks], marker="s",
        color="#264653", label="B trained with noisy weather")
ax.axhline(pooled[24]["predA"]["MAE"], color=COL["A"], ls="--", label="Model A (no forecast weather)")
ax.axhline(pooled[24]["persist"]["MAE"], color=COL["persist"], ls="--", label="Persistence")
ax.set_xlabel("Forecast-weather error (std. dev. as fraction of each variable's std)")
ax.set_ylabel("24h MAE (µg/m³)")
ax.set_title("How robust is model B to imperfect weather forecasts?")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "noise_sensitivity.png"), dpi=150)
plt.close(fig)
log("charts saved")

# --------------------------------------- 8. outputs for the app + final fit
oof = cv_iv[cv_iv["h"].isin(CQR_H)].merge(
    oof_p, on=["fold", "h", "city", "time"], how="left")
oof["time"] = pd.to_datetime(oof["time"])
num = oof.select_dtypes("number").columns
oof[num] = oof[num].round(2)
oof.to_csv(os.path.join(BASE, "backtest_predictions.csv"), index=False)

meta = {"horizons": HORIZONS, "past_cols": PAST_COLS, "fx_cols": FX_COLS,
        "unhealthy": UNHEALTHY, "alpha": ALPHA, "fx_std": STD, "train_noise": NT,
        "trained_until": str(d["time"].max()), "qhat_log": {},
        "clf_threshold": {}, "clf_horizons": CLF_H}
final_cal_start = d["time"].max() - pd.Timedelta(days=90)
for h in HORIZONS:
    XA, valid = frame(h, "A")
    XB, _ = frame(h, "B")
    ok = valid
    joblib.dump({"model": reg().fit(XA[ok], YLOG[h][ok]), "cols": list(XA.columns)},
                os.path.join(MODEL_DIR, f"regA_h{h}.joblib"), compress=3)
    XBn = perturb(XB[ok], NT, seed=31)          # noise-augmented training
    joblib.dump({"model": reg().fit(XBn, YLOG[h][ok]), "cols": list(XB.columns)},
                os.path.join(MODEL_DIR, f"regB_h{h}.joblib"), compress=3)
    # conformal calibration on the last 90 days (models fitted before them)
    fit = ok & (T < final_cal_start - GAP)
    cal = ok & (T >= final_cal_start)
    XBf = perturb(XB, NT, seed=32)
    lo, hi = interval_models(h, fit, XBf)
    yc = YLOG[h][cal].values
    Xc = perturb(XB[cal], NT, seed=33)
    meta["qhat_log"][str(h)] = conformal_q(
        np.maximum(lo.predict(Xc) - yc, yc - hi.predict(Xc)))
    lo, hi = interval_models(h, ok, XBf)          # refit on all data
    joblib.dump({"model": lo, "cols": list(XB.columns)},
                os.path.join(MODEL_DIR, f"q10B_h{h}.joblib"), compress=3)
    joblib.dump({"model": hi, "cols": list(XB.columns)},
                os.path.join(MODEL_DIR, f"q90B_h{h}.joblib"), compress=3)
    if h in CLF_H:
        lab = (Y[h] > UNHEALTHY).astype(int)
        joblib.dump({"model": clf().fit(XBf[ok], lab[ok]), "cols": list(XB.columns)},
                    os.path.join(MODEL_DIR, f"clfB_h{h}.joblib"), compress=3)
        meta["clf_threshold"][str(h)] = clf_res[h]["robust_threshold"]
    log(f"final models h={h} saved")

with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
with open(os.path.join(BASE, "metrics_v3.json"), "w") as f:
    json.dump(metrics, f, indent=2, default=str)
log("DONE")
