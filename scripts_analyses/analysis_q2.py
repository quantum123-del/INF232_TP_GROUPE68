#!/usr/bin/env python3
"""Q2 - Bivariate analysis: correlation and prediction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def analyze_bivariate(dataset_path: str) -> None:
    """Analyze the relationship between study time and evaluation score."""
    df = pd.read_csv(dataset_path)
    x_var = df["temps_etude_heures"]
    y_var = df["score_eval"]

    # Compute correlation coefficients.
    corr_pearson = y_var.corr(x_var)
    corr_spearman = y_var.corr(x_var, method="spearman")

    # Estimate a linear model for prediction.
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_var, y_var)

    model = LinearRegression()
    model.fit(x_var.values.reshape(-1, 1), y_var)
    y_pred = model.predict(x_var.values.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_var, y_pred))
    r_squared = r_value**2

    residuals = y_var - y_pred
    std_residuals = residuals.std()

    results = {
        "correlation_pearson": round(corr_pearson, 3),
        "correlation_spearman": round(corr_spearman, 3),
        "r_squared": round(r_squared, 3),
        "rmse": round(rmse, 2),
        "slope": round(slope, 4),
        "intercept": round(intercept, 2),
        "p_value": round(p_value, 10),
        "std_residuals": round(std_residuals, 2),
    }

    # Print the bivariate analysis summary.
    print("=" * 60)
    print("QUESTION 2 — ANALYSE BIVARIÉE")
    print("Relation entre : Temps d'étude (heures) et Score d'évaluation")
    print("=" * 60)
    print(f"Corrélation de Pearson : {results['correlation_pearson']}")
    print(f"Corrélation de Spearman : {results['correlation_spearman']}")
    print(f"Coefficient de détermination (R²) : {results['r_squared']}")
    print(f"RMSE (erreur moyenne) : {results['rmse']}")
    print(f"P-valeur : {results['p_value']}")
    print(f"Équation de régression : score = {results['intercept']} + {results['slope']} × temps_étude")

    print("\nINTERPRÉTATION :")
    if abs(corr_pearson) < 0.3:
        strength = "faible"
    elif abs(corr_pearson) < 0.7:
        strength = "modérée"
    else:
        strength = "forte"
    direction = "positive" if corr_pearson > 0 else "négative"
    print(f"  Relation {strength} et {direction} entre temps d'étude et score.")
    print(f"  Le temps d'étude explique {results['r_squared'] * 100:.1f}% de la variance du score.")

    if results['p_value'] < 0.05:
        print(f"  Cette relation est statistiquement significative (p < 0.05).")
    else:
        print(f"  Cette relation n'est PAS statistiquement significative (p ≥ 0.05).")

    print("\nFIABILITÉ DE L'ESTIMATION :")
    print(f"  Intervalle de confiance de l'erreur : ±{1.96 * results['std_residuals']:.1f} points")
    print(
        f"  → On peut estimer le score d'un élève à partir de son temps d'étude, "
        f"    mais avec une marge d'erreur de ±{1.96 * results['std_residuals']:.1f} points environ."
    )

    # Visualize the relationship and residuals.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(x_var, y_var, alpha=0.6, s=50, color="steelblue")
    axes[0].plot(x_var, y_pred, "r-", linewidth=2, label=f"Régression (R²={results['r_squared']:.3f})")
    axes[0].set_xlabel("Temps d'étude (heures/semaine)")
    axes[0].set_ylabel("Score d'évaluation")
    axes[0].set_title(f"Relation : Temps d'étude vs Score\n(Corrélation = {corr_pearson:.3f})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].scatter(y_pred, residuals, alpha=0.6, s=50, color="forestgreen")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Valeurs prédites")
    axes[1].set_ylabel("Résidus")
    axes[1].set_title("Diagnostic des résidus")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    output_path = Path(dataset_path).parent.parent / "report" / "q2_bivariate.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"\nGraphiques sauvegardés : {output_path}")

    results_path = output_path.with_suffix(".json")
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nRÉSUMÉ POUR LE PROVISEUR :")
    print("-" * 60)
    print(
        f"Oui, les deux variables évoluent ensemble : plus les élèves étudient, meilleures sont généralement leurs notes. "
        f"Cette relation explique {results['r_squared'] * 100:.1f}% de la variation des scores. "
        f"À l'avenir, on pourrait estimer le score à partir du temps d'étude, "
        f"mais avec une incertitude d'environ ±{1.96 * results['std_residuals']:.1f} points. "
        f"Cette estimation reste fiable si le temps d'étude se situe dans la plage observée (2–25 heures/semaine)."
    )
    plt.close()


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.csv"
    analyze_bivariate(dataset_path)
