"""End-to-end experiment runner for T13."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from config import (HOUSING_FEATURES, IRIS_FEATURES, IRIS_SEEDS, OUT_DIR, SEEDS,
                    TABLE_DIR)
from data_loader import load_clean, run_audit
from detectors import (IForestDetector, build_detector, cv_score, cv_select,
                       default_grid, default_params, evaluate_params,
                       select_threshold, tune_detector)
from evaluation import (aggregate_metrics, binary_metrics, per_type_recall,
                        score_metrics)
from features import (add_features, feature_columns, fit_scaler, select_features,
                      transform)
from inject import inject_anomalies, injection_summary, reference_stats

METHODS = ["Z-Score", "Mahalanobis", "LOF", "Isolation Forest", "One-Class SVM"]
CV_FOLDS = 3
ARTIFACT_DIR = OUT_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def make_splits(n: int, seed: int) -> Dict[str, np.ndarray]:
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=0.30, random_state=seed)
    fit_idx, temp_idx = train_test_split(train_idx, test_size=0.30, random_state=seed + 1)
    cal_idx, val_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed + 2)
    assert len(set(fit_idx) & set(cal_idx)) == 0
    assert len(set(fit_idx) & set(val_idx)) == 0
    assert len(set(fit_idx) & set(test_idx)) == 0
    return {"fit": fit_idx, "cal": cal_idx, "val": val_idx, "test": test_idx}


def _n_anomalies(n: int, fraction: float, minimum: int) -> int:
    return max(minimum, int(round(fraction * n)))


def _type_labels(n: int, type_by_row: Dict[int, str]) -> np.ndarray:
    return np.array([type_by_row.get(i, "normal") for i in range(n)], dtype=object)


def _build_cv_folds(seed: int, raw_fit: pd.DataFrame, stats: Dict,
                    inj_cols: List[str], n_splits: int = CV_FOLDS) -> List[Dict]:
    """K-fold split of the normal training rows, with a per-fold injection.

    Fold frames stay at the raw field level so that different feature subsets can
    reuse the same folds.
    """
    folds = []
    idx = np.arange(len(raw_fit))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed + 7)
    for k, (tr, ho) in enumerate(splitter.split(idx)):
        tr2, cal2 = train_test_split(tr, test_size=0.25, random_state=seed + 8 + k)
        n_inj = _n_anomalies(len(ho), 0.10, 4)
        ho_inj = inject_anomalies(raw_fit.iloc[ho].reset_index(drop=True), n_inj,
                                  seed + 2000 + k, inj_cols, stats, tag="cv")
        folds.append({
            "fold": k,
            "train_idx": tr2,
            "cal_idx": cal2,
            "holdout_frame": ho_inj.frame,
            "y_hold": ho_inj.labels,
            "n_anom": n_inj,
        })
    return folds


def _fold_matrices(p: Dict, cols: Optional[List[str]] = None) -> List[Dict]:
    """Transform the CV folds; the scaler is refitted on each fold's train slice."""
    cols = list(cols if cols is not None else p["selected"])
    raw_fit = p["raw"]["fit"]
    out = []
    for f in p["folds"]:
        train_feat = add_features(raw_fit.iloc[f["train_idx"]].reset_index(drop=True),
                                  p["dataset"])
        cal_feat = add_features(raw_fit.iloc[f["cal_idx"]].reset_index(drop=True),
                                p["dataset"])
        hold_feat = add_features(f["holdout_frame"], p["dataset"])
        scaler = fit_scaler(train_feat[cols], method="robust")
        out.append({
            "X_train": transform(train_feat, scaler, cols),
            "X_cal": transform(cal_feat, scaler, cols),
            "X_hold": transform(hold_feat, scaler, cols),
            "y_hold": f["y_hold"],
        })
    return out


def prepare_seed(dataset: str, seed: int, feature_mode: str = "engineered"):
    df, _ = load_clean(dataset)
    base_cols = IRIS_FEATURES if dataset == "iris" else HOUSING_FEATURES
    splits = make_splits(len(df), seed)
    raw = {k: df.iloc[v].reset_index(drop=True) for k, v in splits.items()}
    stats = reference_stats(raw["fit"], base_cols)
    inj_cols = [c for c in base_cols if c not in ("CHAS", "RAD")] if dataset == "housing" else list(base_cols)

    n_test = _n_anomalies(len(raw["test"]), 0.10, 10 if dataset == "housing" else 4)
    n_val = _n_anomalies(len(raw["val"]), 0.20, 10 if dataset == "housing" else 4)
    test_inj = inject_anomalies(raw["test"], n_test, seed, inj_cols, stats, tag="test")
    val_inj = inject_anomalies(raw["val"], n_val, seed + 1000, inj_cols, stats, tag="val")
    folds = _build_cv_folds(seed, raw["fit"], stats, inj_cols)

    fit_feat = add_features(raw["fit"], dataset)
    cal_feat = add_features(raw["cal"], dataset)
    val_feat = add_features(val_inj.frame, dataset)
    test_feat = add_features(test_inj.frame, dataset)
    cols = feature_columns(dataset, feature_mode)
    selected = select_features(fit_feat, cols, corr_threshold=0.95)
    scaler = fit_scaler(fit_feat[selected], method="robust")
    X = {
        "fit": transform(fit_feat, scaler, selected),
        "cal": transform(cal_feat, scaler, selected),
        "val": transform(val_feat, scaler, selected),
        "test": transform(test_feat, scaler, selected),
    }
    return {
        "dataset": dataset, "df": df, "splits": splits, "raw": raw, "stats": stats,
        "test_injection": test_inj, "val_injection": val_inj, "folds": folds,
        "X": X, "selected": selected, "scaler": scaler,
        "y_val": val_inj.labels, "y_test": test_inj.labels,
        "type_test": _type_labels(len(test_inj.frame), test_inj.type_by_row),
        "n_test_anom": n_test, "n_val_anom": n_val,
    }


def _score_one(model, X: np.ndarray) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    s = model.score(X)
    return s, time.perf_counter() - t0


def run_one_seed(dataset: str, seed: int, feature_mode: str = "engineered",
                 save_artifacts: bool = False) -> Dict:
    t_start = time.perf_counter()
    p = prepare_seed(dataset, seed, feature_mode)
    X, y_val, y_test = p["X"], p["y_val"], p["y_test"]
    folds = _fold_matrices(p)
    records = {}
    tuned_models = []
    curves = {}

    for name in METHODS:
        tuned = tune_detector(name, X["fit"], X["cal"], X["val"], y_val,
                              dataset, seed, folds=folds)
        s_test, score_seconds = _score_one(tuned.model, X["test"])
        tm = binary_metrics(y_test, s_test, tuned.threshold)
        sm = score_metrics(y_test, s_test)
        y_pred = (s_test >= tuned.threshold).astype(int)
        ptr = per_type_recall(p["test_injection"].manifest, y_pred)
        records[name] = {
            "method": name,
            "params": tuned.params,
            "worst_params": tuned.worst_params,
            "threshold": tuned.threshold,
            "fit_seconds": tuned.fit_seconds,
            "score_seconds": score_seconds,
            "val_f1": tuned.val_metrics["f1"],
            "val_recall": tuned.val_metrics["recall"],
            "val_cal_fpr": tuned.val_metrics["cal_fpr"],
            "val_pr_auc": tuned.val_pr_auc,
            "cv_f1": tuned.cv_f1,
            "cv_f1_std": tuned.cv_f1_std,
            "cv_f1_worst": tuned.cv_f1_worst,
            "cv_f1_range": tuned.cv_f1_range,
            "n_candidates": tuned.n_candidates,
            **tm, **sm,
            "per_type_recall": {r.injection_type: float(r.recall) for r in ptr.itertuples()},
        }
        tuned_models.append(tuned)
        curves[name] = s_test

    # Rank-style ensemble over the five tuned detectors (z-score average).
    cal_models = [m.model.score(X["cal"]) for m in tuned_models]
    val_models = [m.model.score(X["val"]) for m in tuned_models]
    test_models = [m.model.score(X["test"]) for m in tuned_models]
    cal_z, val_z, test_z = [], [], []
    for c, v, t in zip(cal_models, val_models, test_models):
        mu, sd = float(np.mean(c)), float(np.std(c, ddof=0))
        sd = sd if sd > 1e-12 else 1.0
        cal_z.append((c - mu) / sd)
        val_z.append((v - mu) / sd)
        test_z.append((t - mu) / sd)
    ens_cal = np.mean(cal_z, axis=0)
    ens_val = np.mean(val_z, axis=0)
    ens_test = np.mean(test_z, axis=0)
    ens_tau, ens_sel = select_threshold(ens_cal, ens_val, y_val)
    ens_tm = binary_metrics(y_test, ens_test, ens_tau)
    ens_sm = score_metrics(y_test, ens_test)
    ens_pred = (ens_test >= ens_tau).astype(int)
    ens_ptr = per_type_recall(p["test_injection"].manifest, ens_pred)
    records["Rank-Ensemble"] = {
        "method": "Rank-Ensemble",
        "params": {"members": METHODS, "aggregation": "z-score mean"},
        "worst_params": {},
        "threshold": ens_tau,
        "fit_seconds": sum(m.fit_seconds for m in tuned_models),
        "score_seconds": sum(m.score_seconds for m in tuned_models),
        "val_f1": ens_sel["f1"],
        "val_recall": ens_sel["recall"],
        "val_cal_fpr": ens_sel["cal_fpr"],
        "val_pr_auc": float(score_metrics(y_val, ens_val)["pr_auc"]),
        "cv_f1": None, "cv_f1_std": None, "cv_f1_worst": None,
        "cv_f1_range": None, "n_candidates": None,
        **ens_tm, **ens_sm,
        "per_type_recall": {r.injection_type: float(r.recall) for r in ens_ptr.itertuples()},
    }
    curves["Rank-Ensemble"] = ens_test

    if save_artifacts:
        manifest = p["test_injection"].manifest.copy()
        manifest["dataset"] = dataset
        manifest["seed"] = seed
        manifest["feature_mode"] = feature_mode
        manifest.to_csv(ARTIFACT_DIR / f"{dataset}_seed{seed}_manifest.csv",
                        index=False, encoding="utf-8-sig")
        safe = {k.replace(" ", "_").replace("-", "_"): v for k, v in curves.items()}
        np.savez_compressed(
            ARTIFACT_DIR / f"{dataset}_seed{seed}_scores.npz",
            X_fit=X["fit"], X_test=X["test"], y_test=y_test,
            type_test=p["type_test"], feature_names=np.array(p["selected"]),
            **{f"score_{k}": v for k, v in safe.items()},
        )

    return {
        "dataset": dataset,
        "seed": seed,
        "feature_mode": feature_mode,
        "n_total": len(p["df"]),
        "n_fit": len(p["splits"]["fit"]),
        "n_cal": len(p["splits"]["cal"]),
        "n_val": len(p["splits"]["val"]),
        "n_test": len(p["splits"]["test"]),
        "n_test_anom": p["n_test_anom"],
        "n_val_anom": p["n_val_anom"],
        "n_cv_folds": len(folds),
        "cv_fold_sizes": [int(len(f["y_hold"])) for f in p["folds"]],
        "cv_fold_anoms": [int(f["n_anom"]) for f in p["folds"]],
        "selected_features": p["selected"],
        "n_features": len(p["selected"]),
        "injection_summary": injection_summary(p["test_injection"].manifest).to_dict("records"),
        "methods": records,
        "elapsed_seconds": time.perf_counter() - t_start,
    }


def run_dataset(dataset: str, feature_mode: str = "engineered",
                seeds: Optional[List[int]] = None) -> Dict:
    seeds = seeds or (IRIS_SEEDS if dataset == "iris" else SEEDS)
    per_seed = []
    for i, seed in enumerate(seeds):
        print(f"[{dataset}] seed {seed} ({i + 1}/{len(seeds)}) ...", flush=True)
        per_seed.append(run_one_seed(dataset, seed, feature_mode, save_artifacts=(i == 0)))
    agg = {}
    for method in METHODS + ["Rank-Ensemble"]:
        recs = [s["methods"][method] for s in per_seed]
        agg[method] = aggregate_metrics(recs)
    return {"per_seed": per_seed, "aggregate": agg}


def performance_table(results: Dict) -> pd.DataFrame:
    rows = []
    for method, agg in results["aggregate"].items():
        def g(k):
            return agg.get(k, {}).get("mean", float("nan"))

        def s(k):
            return agg.get(k, {}).get("std", 0.0)

        rows.append({
            "模型": method,
            "主要参数": _params_text(method, results),
            "PR-AUC": f"{g('pr_auc'):.4f}±{s('pr_auc'):.4f}",
            "Recall": f"{g('recall'):.4f}±{s('recall'):.4f}",
            "Precision": f"{g('precision'):.4f}±{s('precision'):.4f}",
            "F1": f"{g('f1'):.4f}±{s('f1'):.4f}",
            "FPR": f"{g('fpr'):.4f}±{s('fpr'):.4f}",
            "ROC-AUC": f"{g('roc_auc'):.4f}±{s('roc_auc'):.4f}",
            "训练耗时(s)": f"{g('fit_seconds'):.4f}±{s('fit_seconds'):.4f}",
        })
    return pd.DataFrame(rows)


def _params_text(method: str, results: Dict) -> str:
    if method == "Rank-Ensemble":
        return "五模型 z-score 等权平均"
    recs = [s["methods"][method]["params"] for s in results["per_seed"]]
    if not recs[0]:
        return "无超参数"
    counts = Counter(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in recs)
    top, n = counts.most_common(1)[0]
    text = ", ".join(f"{k}={v}" for k, v in json.loads(top).items())
    if n < len(recs):
        text += f"（{n}/{len(recs)} 种子）"
    return text


def per_type_table(results: Dict) -> pd.DataFrame:
    rows = []
    for method, agg in results["aggregate"].items():
        types = ["point", "multi", "extreme"]
        row = {"模型": method}
        for t in types:
            vals = []
            for s in results["per_seed"]:
                v = s["methods"][method]["per_type_recall"].get(t)
                if v is not None:
                    vals.append(v)
            row[{"point": "点异常", "multi": "协同异常", "extreme": "极端组合"}[t]] = (
                f"{np.mean(vals):.3f}" if vals else "-"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run_feature_gain(dataset: str = "housing", seed: int = 42) -> pd.DataFrame:
    """Base vs engineered, plus drop-one marginal contribution of each derived feature."""
    p = prepare_seed(dataset, seed, "engineered")
    X, y_val, y_test = p["X"], p["y_val"], p["y_test"]
    base_cols = feature_columns(dataset, "base")
    eng_cols = feature_columns(dataset, "engineered")
    derived = [c for c in eng_cols if c not in base_cols]

    def evaluate(cols: List[str], params: Optional[Dict] = None):
        selected = select_features(add_features(p["raw"]["fit"], dataset), cols)
        scaler = fit_scaler(add_features(p["raw"]["fit"], dataset)[selected], "robust")
        Xf = transform(add_features(p["raw"]["fit"], dataset), scaler, selected)
        Xc = transform(add_features(p["raw"]["cal"], dataset), scaler, selected)
        Xv = transform(add_features(p["val_injection"].frame, dataset), scaler, selected)
        Xt = transform(add_features(p["test_injection"].frame, dataset), scaler, selected)
        if params is None:
            tuned = tune_detector("Isolation Forest", Xf, Xc, Xv, y_val,
                                  dataset, seed,
                                  grid=default_grid(dataset, "Isolation Forest"),
                                  folds=_fold_matrices(p, selected))
            model, tau = tuned.model, tuned.threshold
            val_f1, val_pr = tuned.val_metrics["f1"], tuned.val_pr_auc
        else:
            model = IForestDetector(random_state=seed, **params).fit(Xf)
            s_c, s_v = model.score(Xc), model.score(Xv)
            tau, sel = select_threshold(s_c, s_v, y_val)
            val_f1, val_pr = sel["f1"], score_metrics(y_val, s_v)["pr_auc"]
        s_t = model.score(Xt)
        tm = binary_metrics(y_test, s_t, tau)
        sm = score_metrics(y_test, s_t)
        return {
            "n_features": len(selected), "val_f1": val_f1, "val_pr_auc": val_pr,
            "pr_auc": sm["pr_auc"], "recall": tm["recall"], "f1": tm["f1"],
            "fpr": tm["fpr"], "threshold": tau, "model": model, "selected": selected,
        }

    base = evaluate(base_cols)
    full = evaluate(eng_cols)
    full_params = {k: v for k, v in full.get("params", {}).items()} if "params" in full else None
    # Use the tuned full-set parameters for the drop-one comparison.
    # Recover them from the model object when available.
    if hasattr(full["model"], "n_estimators"):
        full_params = {
            "n_estimators": full["model"].n_estimators,
            "max_samples": full["model"].max_samples,
            "max_features": full["model"].max_features,
        }
    else:
        full_params = {"n_estimators": 200, "max_samples": 1.0, "max_features": 1.0}

    rows = [{
        "特征集": "基础特征", "特征数": base["n_features"],
        "PR-AUC": base["pr_auc"], "Recall": base["recall"], "F1": base["f1"],
        "FPR": base["fpr"], "说明": "原始字段",
    }, {
        "特征集": "工程特征", "特征数": full["n_features"],
        "PR-AUC": full["pr_auc"], "Recall": full["recall"], "F1": full["f1"],
        "FPR": full["fpr"], "说明": "原始字段+派生特征",
    }]
    for f in derived:
        if f not in full["selected"]:
            continue
        cols = [c for c in full["selected"] if c != f]
        r = evaluate(cols, params=full_params)
        rows.append({
            "特征集": f"去掉 {f}", "特征数": r["n_features"],
            "PR-AUC": r["pr_auc"], "Recall": r["recall"], "F1": r["f1"],
            "FPR": r["fpr"], "说明": f"边际贡献 ΔPR-AUC={full['pr_auc'] - r['pr_auc']:+.4f}",
        })
    return pd.DataFrame(rows)


def run_sensitivity(dataset: str = "housing", seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    p = prepare_seed(dataset, seed, "engineered")
    X, y_val, y_test = p["X"], p["y_val"], p["y_test"]
    qs = [0.90, 0.925, 0.95, 0.975, 0.99]
    rows = []

    def add(method, param_name, param_value, q, s_cal, s_test):
        tau = float(np.quantile(s_cal, q))
        tm = binary_metrics(y_test, s_test, tau)
        rows.append({"method": method, "param_name": param_name, "param_value": str(param_value),
                     "threshold_quantile": q, "pr_auc": score_metrics(y_test, s_test)["pr_auc"],
                     "f1": tm["f1"], "recall": tm["recall"], "fpr": tm["fpr"]})

    for k in (5, 10, 15, 20, 30, 50):
        if k >= len(X["fit"]):
            continue
        m = build_detector("LOF", {"n_neighbors": k}, seed).fit(X["fit"])
        s_cal, s_test = m.score(X["cal"]), m.score(X["test"])
        for q in qs:
            add("LOF", "n_neighbors", k, q, s_cal, s_test)

    for ms in (0.3, 0.5, 0.7, 0.85, 1.0):
        m = build_detector("Isolation Forest",
                           {"n_estimators": 200, "max_samples": ms, "max_features": 1.0},
                           seed).fit(X["fit"])
        s_cal, s_test = m.score(X["cal"]), m.score(X["test"])
        for q in qs:
            add("Isolation Forest", "max_samples", ms, q, s_cal, s_test)

    for nu in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
        for gamma in ("scale", 0.01, 0.05, 0.1, 0.5):
            m = build_detector("One-Class SVM", {"nu": nu, "gamma": gamma}, seed).fit(X["fit"])
            s_cal, s_test = m.score(X["cal"]), m.score(X["test"])
            add("One-Class SVM", "gamma", f"nu={nu},gamma={gamma}", 0.95, s_cal, s_test)

    for sf in (0.5, 0.6, 0.75, 0.9, 1.0):
        m = build_detector("Mahalanobis", {"support_fraction": sf}, seed).fit(X["fit"])
        s_cal, s_test = m.score(X["cal"]), m.score(X["test"])
        for q in qs:
            add("Mahalanobis", "support_fraction", sf, q, s_cal, s_test)

    m = build_detector("Z-Score", {}, seed).fit(X["fit"])
    s_cal, s_test = m.score(X["cal"]), m.score(X["test"])
    for q in qs:
        add("Z-Score", "threshold_quantile", "-", q, s_cal, s_test)

    long = pd.DataFrame(rows)
    summary_rows = []
    for method, grp in long.groupby("method"):
        summary_rows.append({
            "method": method,
            "min_f1": grp["f1"].min(),
            "max_f1": grp["f1"].max(),
            "f1_range": grp["f1"].max() - grp["f1"].min(),
            "max_pr_auc": grp["pr_auc"].max(),
        })
    return long, pd.DataFrame(summary_rows).sort_values("f1_range", ascending=False)


TUNED_SETTINGS = ("默认参数", "交叉验证最优", "交叉验证最差")
SETTING_ORDER = {s: i for i, s in enumerate(TUNED_SETTINGS + ("无超参数",))}


def _short_params(params: Dict) -> str:
    if not params:
        return "无超参数"
    return ", ".join(f"{k}={v}" for k, v in params.items())


def run_param_compare(dataset: str = "housing",
                      seeds: Optional[List[int]] = None) -> pd.DataFrame:
    """Compare sklearn defaults, the CV-selected setting and the worst CV setting.

    All three are fitted on the same normal rows, given the same threshold policy
    and scored on the same test rows, so only the hyper-parameter setting varies.
    """
    seeds = seeds or (IRIS_SEEDS if dataset == "iris" else SEEDS)
    rows = []
    for seed in seeds:
        p = prepare_seed(dataset, seed, "engineered")
        folds = _fold_matrices(p)
        X, y_val, y_test = p["X"], p["y_val"], p["y_test"]
        for name in METHODS:
            if name == "Z-Score":
                settings = {"无超参数": {}}
                cv_f1 = {"无超参数": float("nan")}
            else:
                best, worst, _ = cv_select(name, folds, dataset, seed,
                                           default_grid(dataset, name))
                settings = {
                    "默认参数": default_params(name),
                    "交叉验证最优": best["params"],
                    "交叉验证最差": worst["params"],
                }
                cv_f1 = {
                    "默认参数": cv_score(name, settings["默认参数"], folds, seed)["cv_f1_mean"],
                    "交叉验证最优": best["cv_f1_mean"],
                    "交叉验证最差": worst["cv_f1_mean"],
                }
            for setting, params in settings.items():
                tuned = evaluate_params(name, params, X["fit"], X["cal"],
                                        X["val"], y_val, seed)
                s_test = tuned.model.score(X["test"])
                tm = binary_metrics(y_test, s_test, tuned.threshold)
                sm = score_metrics(y_test, s_test)
                rows.append({
                    "dataset": dataset, "seed": seed, "method": name,
                    "setting": setting, "params": _short_params(params),
                    "cv_f1": cv_f1[setting],
                    "recall": tm["recall"], "precision": tm["precision"],
                    "f1": tm["f1"], "fpr": tm["fpr"], "pr_auc": sm["pr_auc"],
                })
    return pd.DataFrame(rows)


def param_compare_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the parameter settings over seeds for the report."""
    out = []
    for method in METHODS:
        g = raw[raw["method"] == method]
        if g.empty:
            continue
        for setting in sorted(g["setting"].unique(), key=lambda s: SETTING_ORDER[s]):
            sub = g[g["setting"] == setting]
            counts = Counter(sub["params"])
            params, n = counts.most_common(1)[0]
            if n < len(sub):
                params = f"{params}（{n}/{len(sub)} 种子）"
            cv = sub["cv_f1"].mean()
            gap = sub["f1"].mean() - cv if np.isfinite(cv) else float("nan")
            out.append({
                "模型": method, "参数设置": setting, "参数取值": params,
                "CV F1": "-" if not np.isfinite(cv) else f"{cv:.4f}",
                "测试 F1": f"{sub['f1'].mean():.4f}",
                "CV-测试差": "-" if not np.isfinite(gap) else f"{gap:+.4f}",
                "Recall": f"{sub['recall'].mean():.4f}",
                "FPR": f"{sub['fpr'].mean():.4f}",
            })
    return pd.DataFrame(out)


def param_stability_table(results: Dict) -> pd.DataFrame:
    """Per-seed record of the CV-selected setting, for the appendix."""
    rows = []
    for s in results["per_seed"]:
        for name in METHODS:
            r = s["methods"][name]
            if r.get("cv_f1") is None:
                continue
            rows.append({
                "模型": name, "种子": s["seed"],
                "选中的参数": _short_params(r["params"]),
                "CV F1": f"{r['cv_f1']:.4f}",
                "CV 极差": f"{r['cv_f1_range']:.4f}",
                "测试 F1": f"{r['f1']:.4f}",
            })
    return pd.DataFrame(rows)


def run_main() -> Dict:
    run_audit()
    all_results = {}
    for dataset in ("housing", "iris"):
        res = run_dataset(dataset, "engineered")
        all_results[dataset] = res
        performance_table(res).to_csv(TABLE_DIR / f"table_performance_{dataset}.csv",
                                      index=False, encoding="utf-8-sig")
        per_type_table(res).to_csv(TABLE_DIR / f"table_per_type_{dataset}.csv",
                                   index=False, encoding="utf-8-sig")

    # Combined performance table for the report (housing main, iris appendix).
    perf = performance_table(all_results["housing"])
    perf.insert(0, "数据集", "housing")
    iris_perf = performance_table(all_results["iris"])
    iris_perf.insert(0, "数据集", "iris")
    pd.concat([perf, iris_perf], ignore_index=True).to_csv(
        TABLE_DIR / "table_model_performance.csv", index=False, encoding="utf-8-sig")

    gain = run_feature_gain("housing", 42)
    gain.to_csv(TABLE_DIR / "table_feature_gain.csv", index=False, encoding="utf-8-sig")

    sens_long, sens_summary = run_sensitivity("housing", 42)
    sens_long.to_csv(TABLE_DIR / "table_sensitivity.csv", index=False, encoding="utf-8-sig")
    sens_summary.to_csv(TABLE_DIR / "table_sensitivity_summary.csv",
                        index=False, encoding="utf-8-sig")

    # Tuning necessity: sklearn defaults vs CV-selected vs worst-in-grid.
    pc_raw = run_param_compare("housing", SEEDS)
    pc_raw.to_csv(TABLE_DIR / "table_param_compare_raw.csv", index=False,
                  encoding="utf-8-sig")
    param_compare_table(pc_raw).to_csv(TABLE_DIR / "table_param_compare.csv",
                                       index=False, encoding="utf-8-sig")
    param_stability_table(all_results["housing"]).to_csv(
        TABLE_DIR / "table_param_stability.csv", index=False, encoding="utf-8-sig")

    # Injection configuration table.
    inj_rows = []
    for dataset, res in all_results.items():
        s = res["per_seed"][0]
        inj_rows.append({
            "数据集": dataset, "总样本": s["n_total"], "训练": s["n_fit"],
            "校准": s["n_cal"], "验证": s["n_val"], "测试": s["n_test"],
            "测试注入异常": s["n_test_anom"], "验证注入异常": s["n_val_anom"],
            "异常占比": f"{s['n_test_anom'] / s['n_test'] * 100:.2f}%",
        })
    pd.DataFrame(inj_rows).to_csv(TABLE_DIR / "table_injection_config.csv",
                                  index=False, encoding="utf-8-sig")

    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    with open(OUT_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump({k: v["aggregate"] for k, v in all_results.items()},
                  f, ensure_ascii=False, indent=2, default=str)
    print("main experiments complete")
    print(performance_table(all_results["housing"]).to_string(index=False))
    return all_results


if __name__ == "__main__":
    run_main()
