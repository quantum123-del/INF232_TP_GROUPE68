#!/usr/bin/env python3
"""Q3 - Unsupervised clustering: discovering natural student profiles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler


def analyze_clustering(dataset_path: str) -> None:
    """Perform unsupervised clustering to discover student profiles."""
    df = pd.read_csv(dataset_path)
    X = df[["score_eval", "temps_etude_heures", "assiduite_pct"]].values

    # Standardize data to give each feature equal weight in clustering.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    silhouette_scores = []
    k_range = range(2, 11)

    # Evaluate clustering quality for several cluster counts.
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(sil_score)

    optimal_k = k_range[np.argmax(silhouette_scores)]

    # Fit the final KMeans model using the optimal cluster count.
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans_final.fit_predict(X_scaled)
    df["cluster"] = labels

    results = {
        "nombre_clusters": optimal_k,
        "silhouette_score": round(max(silhouette_scores), 3),
        "clusters": {},
    }

    print("=" * 60)
    print("QUESTION 3 — CLASSIFICATION NON SUPERVISÉE (CLUSTERING)")
    print("Découverte de profils naturels d'élèves")
    print("=" * 60)
    print(f"Nombre optimal de profils identifiés : {optimal_k}")
    print(f"Score de silhouette : {results['silhouette_score']} (plage : -1 à 1, >0.5 = bon)")
    print()

    # Describe each cluster in plain language.
    for cluster_id in range(optimal_k):
        cluster_data = df[df["cluster"] == cluster_id]
        mean_score = cluster_data["score_eval"].mean()
        mean_temps = cluster_data["temps_etude_heures"].mean()
        mean_assiduite = cluster_data["assiduite_pct"].mean()
        size = len(cluster_data)

        results["clusters"][f"profil_{cluster_id + 1}"] = {
            "taille": size,
            "pct_effectif": round(100 * size / len(df), 1),
            "score_eval_moyen": round(mean_score, 1),
            "temps_etude_moyen": round(mean_temps, 1),
            "assiduite_moyenne": round(mean_assiduite, 1),
        }

        print(f"PROFIL {cluster_id + 1} ({size} élèves, {100 * size / len(df):.1f}%)")
        print(f"  Score moyen : {mean_score:.1f}/100")
        print(f"  Temps d'étude moyen : {mean_temps:.1f} h/semaine")
        print(f"  Assiduité moyenne : {mean_assiduite:.1f}%")

        if mean_score > 75 and mean_assiduite > 75:
            desc = "Élèves excellents, très assidus et travailleurs."
        elif mean_score > 60 and mean_temps > 10:
            desc = "Élèves performants et studieux."
        elif mean_score < 50 or mean_assiduite < 50:
            desc = "Élèves en difficulté, ayant besoin d'accompagnement."
        else:
            desc = "Élèves moyens, avec une implication variable."
        print(f"  Description : {desc}")
        print()

    # Create visualizations for cluster evaluation and distribution.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
    axes[0, 0].set_xlabel("Nombre de clusters (k)")
    axes[0, 0].set_ylabel("Inertie")
    axes[0, 0].set_title("Coude (Elbow Method)")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(k_range, silhouette_scores, "go-", linewidth=2, markersize=8)
    axes[0, 1].axvline(optimal_k, color="red", linestyle="--", label=f"Optimal k={optimal_k}")
    axes[0, 1].set_xlabel("Nombre de clusters (k)")
    axes[0, 1].set_ylabel("Score de silhouette")
    axes[0, 1].set_title("Qualité du clustering (Silhouette)")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    scatter1 = axes[1, 0].scatter(
        df["score_eval"], df["temps_etude_heures"], c=labels, cmap="viridis", s=80, alpha=0.6
    )
    axes[1, 0].set_xlabel("Score d'évaluation")
    axes[1, 0].set_ylabel("Temps d'étude (h/semaine)")
    axes[1, 0].set_title("Profils par Score vs Temps d'étude")
    plt.colorbar(scatter1, ax=axes[1, 0], label="Cluster")

    scatter2 = axes[1, 1].scatter(
        df["score_eval"], df["assiduite_pct"], c=labels, cmap="viridis", s=80, alpha=0.6
    )
    axes[1, 1].set_xlabel("Score d'évaluation")
    axes[1, 1].set_ylabel("Assiduité (%)")
    axes[1, 1].set_title("Profils par Score vs Assiduité")
    plt.colorbar(scatter2, ax=axes[1, 1], label="Cluster")

    fig.tight_layout()
    output_path = Path(dataset_path).parent.parent / "report" / "q3_clustering.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Graphiques sauvegardés : {output_path}")

    results_path = output_path.with_suffix(".json")
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nRÉSUMÉ POUR LE CONSEIL PÉDAGOGIQUE :")
    print("-" * 60)
    print(
        f"L'analyse a identifié {optimal_k} profils naturels et distincts dans vos données. "
        f"Ces groupes correspondent à des élèves aux trajectoires et besoins différents. "
        f"Chaque profil pourrait bénéficier d'actions d'accompagnement spécifiques : "
        f"approfondissement pour les excellents, soutien pour les en difficulté, etc."
    )
    plt.close()


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.csv"
    analyze_clustering(dataset_path)
