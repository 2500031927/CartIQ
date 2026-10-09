"""RFM customer segmentation (M4): K-means(++), elbow/silhouette, DBSCAN, PCA."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import DBSCAN, KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # noqa: E402

from src import config  # noqa: E402
from src.evaluation.evaluate import save_json_report  # noqa: E402
from src.features.preprocessing import build_customer_snapshot  # noqa: E402
from src.monitoring.monitor import make_version, write_model_metadata  # noqa: E402

log = logging.getLogger(__name__)
RFM = ["recency_days", "frequency", "monetary"]
MODEL_FILE = "customer_cluster_model.joblib"

ACTIONS = {
    "Loyal high-value": "Reward and retain: early access, loyalty perks, referral offers.",
    "At-risk former high-value": "Win-back: personalised outreach and a targeted incentive before they lapse for good.",
    "Recent occasional": "Nurture: second-purchase nudges, bundles and product recommendations.",
    "Lapsed low-value": "Low-cost reactivation only (e.g. email); avoid expensive discounts.",
    "Core regulars": "Maintain: cross-sell and seasonal campaigns.",
}


def _rfm_pipeline(model) -> Pipeline:
    # log1p tames the heavy right skew of F and M; scaling puts R, F, M on equal footing for distance-based K-means
    return Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                     ("scale", StandardScaler()), ("kmeans", model)])


def name_clusters(profile: pd.DataFrame, overall: pd.DataFrame) -> dict[int, str]:
    """Human-readable names derived from MEASURED cluster medians vs the overall distribution."""
    q = overall[RFM].quantile([0.5, 0.75])
    names: dict[int, str] = {}
    for cid, row in profile.iterrows():
        high_value = row["monetary_median"] >= q.loc[0.75, "monetary"]
        low_value = row["monetary_median"] <= q.loc[0.5, "monetary"]
        recent = row["recency_median"] <= q.loc[0.5, "recency_days"]
        if high_value and recent:
            n = "Loyal high-value"
        elif high_value:
            n = "At-risk former high-value"
        elif low_value and recent:
            n = "Recent occasional"
        elif low_value:
            n = "Lapsed low-value"
        else:
            n = "Core regulars"
        names[int(cid)] = n
    seen: dict[str, int] = {}
    for cid, n in list(names.items()):  # keep names unique
        seen[n] = seen.get(n, 0) + 1
        if list(names.values()).count(n) > 1:
            names[cid] = f"{n} (cluster {cid})"
    return names


def train_segmentation(lines: pd.DataFrame, artifact_dir: Path | str = config.ARTIFACT_DIR,
                       k_range=range(3, 9)) -> dict:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rdir = config.report_dir(artifact_dir)
    # Snapshot "as of" just after the last transaction: describes the current customer base.
    as_of = lines["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    snap = build_customer_snapshot(lines, as_of)
    if len(snap) < 30:
        raise ValueError(f"Only {len(snap)} customers - too few for segmentation.")
    X = snap[RFM]
    Xs = _rfm_pipeline(KMeans(n_clusters=2)).named_steps["log"].fit_transform(X)
    Xs = StandardScaler().fit_transform(Xs)

    k_stats = []
    for k in k_range:
        if k >= len(snap):
            continue
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=config.RANDOM_STATE).fit(Xs)
        sil = silhouette_score(Xs, km.labels_, sample_size=min(5000, len(Xs)), random_state=config.RANDOM_STATE)
        k_stats.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})
    best_k = max(k_stats, key=lambda d: d["silhouette"])["k"]

    pipe = _rfm_pipeline(KMeans(n_clusters=best_k, init="k-means++", n_init=10,
                                random_state=config.RANDOM_STATE)).fit(X)
    labels = pipe.predict(X)
    snap = snap.assign(cluster=labels)

    profile = snap.groupby("cluster").agg(
        customers=("CustomerID", "size"), recency_median=("recency_days", "median"),
        frequency_median=("frequency", "median"), monetary_median=("monetary", "median"),
        monetary_total=("monetary", "sum"))
    profile["customer_share"] = profile["customers"] / len(snap)
    profile["revenue_share"] = profile["monetary_total"] / snap["monetary"].sum()
    names = name_clusters(profile, snap)
    profile["name"] = [names[int(c)] for c in profile.index]

    # DBSCAN: eps from the 5-NN distance distribution (density-based comparison; finds noise points)
    nn = NearestNeighbors(n_neighbors=6).fit(Xs)
    eps = float(np.percentile(nn.kneighbors(Xs)[0][:, -1], 90))
    db = DBSCAN(eps=eps, min_samples=10).fit(Xs)
    n_db = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    non_noise = db.labels_ != -1
    db_sil = None
    if n_db >= 2 and non_noise.sum() > n_db:
        db_sil = float(silhouette_score(Xs[non_noise], db.labels_[non_noise]))

    # plots: elbow, silhouette, PCA scatter
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ks = [d["k"] for d in k_stats]
    axes[0].plot(ks, [d["inertia"] for d in k_stats], "o-")
    axes[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
    axes[1].plot(ks, [d["silhouette"] for d in k_stats], "o-")
    axes[1].set(title="Silhouette score", xlabel="k", ylabel="Silhouette")
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    Z = pca.fit_transform(Xs)
    sc = axes[2].scatter(Z[:, 0], Z[:, 1], c=labels, s=6, cmap="tab10")
    axes[2].set(title=f"PCA view of K-means (k={best_k})", xlabel="PC1", ylabel="PC2")
    fig.colorbar(sc, ax=axes[2])
    fig.tight_layout()
    fig.savefig(rdir / "segmentation_plots.png", dpi=120)
    plt.close(fig)

    version = make_version()
    bundle = {"pipeline": pipe, "feature_columns": RFM, "k": best_k,
              "cluster_names": names, "version": version, "as_of": str(as_of),
              "cluster_profile": profile.reset_index().to_dict(orient="records")}
    joblib.dump(bundle, artifact_dir / MODEL_FILE)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    seg_path = Path(artifact_dir) / "customer_segments.csv"
    snap[["CustomerID"] + RFM + ["cluster"]].assign(segment=snap["cluster"].map(names)).to_csv(seg_path, index=False)

    report = {"as_of": str(as_of), "customers": len(snap), "chosen_k": best_k,
              "k_selection": k_stats, "selection_rule": "highest silhouette among k in range (k>=3 for usefulness)",
              "cluster_profile": profile.reset_index().to_dict(orient="records"),
              "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
              "dbscan": {"eps": eps, "min_samples": 10, "clusters": n_db,
                         "noise_fraction": float((~non_noise).mean()), "silhouette_non_noise": db_sil},
              "limitations": ["Clusters describe behaviour as of the last transaction; they drift over time.",
                              "Names come from rule-based comparison to overall medians, not ground truth.",
                              "Only 3 behavioural variables are used; no demographics or product data."]}
    save_json_report(report, rdir / "segmentation_report.json")
    write_model_metadata(artifact_dir, "segmentation", {
        "version": version, "k": best_k, "trained_at": datetime.now(timezone.utc).isoformat(),
        "silhouette": next(d["silhouette"] for d in k_stats if d["k"] == best_k)})
    return report


def predict_segment(bundle: dict, recency_days: float, frequency: float, monetary: float) -> dict:
    X = pd.DataFrame([{"recency_days": recency_days, "frequency": frequency, "monetary": monetary}])
    pipe: Pipeline = bundle["pipeline"]
    cid = int(pipe.predict(X)[0])
    centers = pipe.named_steps["kmeans"].cluster_centers_
    z = pipe[:-1].transform(X)[0]
    name = bundle["cluster_names"][cid]
    base = name.split(" (cluster")[0]
    return {"cluster_id": cid, "segment_name": name,
            "recommended_action": ACTIONS.get(base, ACTIONS["Core regulars"]),
            "distance_to_centroid": float(np.linalg.norm(z - centers[cid]))}
