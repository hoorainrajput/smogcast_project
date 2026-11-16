# Builds site/data/*.json and copies charts from the trained models/metrics.
import json, os, shutil
import joblib, numpy as np, pandas as pd
from features import CITIES, PAST_COLS, FX_COLS, build_past_features, future_weather
B = os.path.dirname(os.path.abspath(__file__)); SITE = os.environ.get("SITE", os.path.join(B, "..", "site"))
meta = json.load(open(f"{B}/models_v3/meta.json")); HZ = meta["horizons"]
raw = pd.read_csv(f"{B}/data/pakistan_air_quality_hourly.csv", parse_dates=["time"])
d = build_past_features(raw)
last = d["time"].max(); origin = last - pd.Timedelta(hours=24)   # replay: all 24 targets are known
ld = lambda k, h: joblib.load(f"{B}/models_v3/{k}_h{h}.joblib")
out = {"issued": str(origin)[:16], "cities": {}}
for c in CITIES:
    idx = d.index[(d.city == c) & (d.time == origin)][0]
    ps, los, his, pe = {}, {}, {}, {}
    for h in HZ:
        fx = future_weather(d, h)
        X = pd.concat([d[PAST_COLS], fx], axis=1).loc[[idx]]
        ps[h] = float(np.clip(np.expm1(ld("regB", h)["model"].predict(X)), 0, None)[0])
        q = meta["qhat_log"][str(h)]
        los[h] = float(max(np.expm1(ld("q10B", h)["model"].predict(X)[0] - q), 0))
        his[h] = float(np.expm1(ld("q90B", h)["model"].predict(X)[0] + q))
        if str(h) in meta["clf_threshold"]:
            pe[h] = float(ld("clfB", h)["model"].predict_proba(X)[0, 1])
    hs = np.arange(1, 25)
    P, L, H = (np.interp(hs, HZ, [m[h] for h in HZ]) for m in (ps, los, his))
    L = np.minimum(L, P); H = np.maximum(H, P)
    act = d[(d.city == c)].set_index("time")["pm2_5"]
    out["cities"][c] = {"now": float(act[origin]), "p_exceed": {str(h): round(p, 3) for h, p in pe.items()},
        "hours": [{"t": str(origin + pd.Timedelta(hours=int(h)))[:16], "pred": round(float(P[i]), 1), "lo": round(float(L[i]), 1),
                   "hi": round(float(H[i]), 1), "actual": round(float(act[origin + pd.Timedelta(hours=int(h))]), 1)} for i, h in enumerate(hs)]}
os.makedirs(f"{SITE}/data", exist_ok=True); os.makedirs(f"{SITE}/assets", exist_ok=True)
json.dump(out, open(f"{SITE}/data/forecast.json", "w"))
M = json.load(open(f"{B}/metrics_v3.json"))
for h, r in M["pooled_cv"].items():
    r["skill_B_vs_persistence_pct"] = r.pop("skill_B_vs_persistence_%"); r.pop("skill_B_vs_best_baseline_%", None)
json.dump({k: M[k] for k in ("horizons", "folds", "pooled_cv", "episode_classifier", "bootstrap_MAE_gain")}, open(f"{SITE}/data/metrics.json", "w"))
for f in ("mae_vs_horizon", "cv_folds_h24", "ablation", "noise_sensitivity", "interval_example", "interval_coverage"):
    shutil.copy(f"{B}/charts_v3/{f}.png", f"{SITE}/assets/{f}.png")
print("site data built; origin", origin)
