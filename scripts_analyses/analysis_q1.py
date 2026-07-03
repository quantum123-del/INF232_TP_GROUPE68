#!/usr/bin/env python3
"""Q1 - Univariate analysis: distribution and outliers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def analyze_univariate(dataset_path: str) -> None:
    """Perform a univariate descriptive analysis on the evaluation score."""
    df = pd.read_csv(dataset_path)
    score_eval = df["score_eval"]

    # Compute central tendency and dispersion measures.
    mean_val = score_eval.mean()
    median_val = score_eval.median()
    std_val = score_eval.std()
    q25_val = score_eval.quantile(0.25)
    q75_val = score_eval.quantile(0.75)
    iqr = q75_val - q25_val

    # Compute shape measures: skewness and kurtosis.
    skewness = stats.skew(score_eval)
    kurtosis = stats.kurtosis(score_eval)

    # Detect outliers using IQR and z-score methods.
    lower_bound = q25_val - 1.5 * iqr
    upper_bound = q75_val + 1.5 * iqr
    outliers = df[(score_eval < lower_bound) | (score_eval > upper_bound)]

    z_scores = np.abs(stats.zscore(score_eval))
    z_outliers = df[z_scores > 3]

    results = {
        "mesure": "score_eval",
        "effectif": len(score_eval),
        "moyenne": round(mean_val, 2),
        "mediane": round(median_val, 2),
        "ecart_type": round(std_val, 2),
        "min": round(score_eval.min(), 2),
        "max": round(score_eval.max(), 2),
        "q25": round(q25_val, 2),
        "q75": round(q75_val, 2),
        "iqr": round(iqr, 2),
        "skewness": round(skewness, 2),
        "kurtosis": round(kurtosis, 2),
        "outliers_iqr": len(outliers),
        "outliers_zscore": len(z_outliers),
    }

    # Print the analysis summary.
    print("=" * 60)
    print("QUESTION 1 — ANALYSE UNIVARIÉE : Score d'évaluation")
    print("=" * 60)
    print(f"Effectif : {results['effectif']} élèves")
    print(f"Moyenne : {results['moyenne']} / 100")
    print(f"Médiane : {results['mediane']} / 100")
    print(f"Écart-type : {results['ecart_type']}")
    print(f"Plage (min–max) : [{results['min']}, {results['max']}]")
    print(f"Quartiles Q1–Q3 : [{results['q25']}, {results['q75']}]")
    print(f"Asymétrie (skewness) : {results['skewness']}")

    print(f"  → Interprétation : ", end="")
    if abs(results['skewness']) < 0.5:
        print("Distribution symétrique")
    elif results['skewness'] > 0:
        print("Queue vers la droite (meilleurs résultats)")
    else:
        print("Queue vers la gauche (résultats plus faibles)")

    print(f"Valeurs atypiques détectées (IQR) : {results['outliers_iqr']}")
    if len(outliers) > 0:
        print("  Élèves concernés :")
        for idx, row in outliers.iterrows():
            print(f"    - ID {row['id']}: score = {row['score_eval']}")

    # Build visualizations: histogram and boxplot.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(score_eval, bins=20, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Moyenne: {mean_val:.1f}")
    axes[0].axvline(median_val, color="green", linestyle="--", linewidth=2, label=f"Médiane: {median_val:.1f}")
    axes[0].set_xlabel("Score d'évaluation")
    axes[0].set_ylabel("Fréquence")
    axes[0].set_title("Distribution des scores d'évaluation")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].boxplot(score_eval, vert=True)
    axes[1].set_ylabel("Score d'évaluation")
    axes[1].set_title("Boîte à moustaches des scores")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    output_path = Path(dataset_path).parent.parent / "report" / "q1_univariate.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"\nGraphique sauvegardé : {output_path}")

    results_path = output_path.with_suffix(".json")
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nRÉSUMÉ POUR LE PROVISEUR (langage simple) :")
    print("-" * 60)
    print(
        f"En moyenne, vos élèves obtiennent une note de {results['moyenne']} sur 100. "
        f"La moitié d'entre eux ont une note au-dessus de {results['mediane']}, et l'autre moitié en dessous. "
        f"Les résultats varient de {results['min']} à {results['max']}. "
        f"Nous avons identifié {results['outliers_iqr']} élèves dont les résultats s'écartent significativement "
        f"de la tendance générale : ces cas méritent peut-être un suivi particulier."
    )
    plt.close()


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.csv"
    analyze_univariate(dataset_path)
