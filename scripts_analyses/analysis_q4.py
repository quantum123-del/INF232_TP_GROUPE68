#!/usr/bin/env python3
"""Q4 - Supervised classification: predicting orientation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def analyze_classification(dataset_path: str) -> None:
    """Train and compare supervised models for orientation prediction."""
    df = pd.read_csv(dataset_path)
    X = df[["score_eval", "temps_etude_heures", "assiduite_pct"]].values
    y = (df["orientation_recommandee"] == "scientifique").astype(int).values

    # Standardize the features for model training.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Use only allowed supervised classifiers: centroid-approach (implemented via KMeans centroids + nearest centroid),
    # k-NN, linear models and SVM. We'll provide LogisticRegression (linear), SVC (linear kernel) and KNeighborsClassifier.
    models = {
        "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
        "svm_linear": SVC(kernel="linear", probability=True, random_state=42),
        "k_nearest": KNeighborsClassifier(n_neighbors=5),
    }

    results = {"models": {}}

    print("=" * 60)
    print("QUESTION 4 — CLASSIFICATION SUPERVISÉE")
    print("Prédiction de l'orientation recommandée")
    print("=" * 60)
    print("Évaluation par validation croisée (5 plis) :")
    print()

    best_model_name = None
    best_score = -1

    for model_name, model in models.items():
        # Evaluate model with stratified cross-validation.
        cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="roc_auc")
        mean_cv_score = cv_scores.mean()

        # Train on the full dataset for final metrics.
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        # Some estimators (SVC) may provide `predict_proba` when probability=True, otherwise use decision_function.
        try:
            y_proba = model.predict_proba(X_scaled)[:, 1]
        except Exception:
            # fallback to decision_function scaled to [0,1]
            df_scores = model.decision_function(X_scaled)
            y_proba = (df_scores - df_scores.min()) / (df_scores.max() - df_scores.min()) if df_scores.max() > df_scores.min() else np.zeros_like(df_scores)

        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        auc_score = roc_auc_score(y, y_proba)

        results["models"][model_name] = {
            "cv_auc_mean": round(mean_cv_score, 3),
            "cv_auc_std": round(cv_scores.std(), 3),
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "auc": round(auc_score, 3),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }

        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_model_name = model_name

        print(f"{model_name.upper()}")
        print(f"  CV AUC : {mean_cv_score:.3f} (±{cv_scores.std():.3f})")
        print(f"  Accuracy : {accuracy:.3f}")
        print(f"  Precision : {precision:.3f}  |  Recall : {recall:.3f}  |  F1 : {f1:.3f}")
        print(f"  AUC-ROC : {auc_score:.3f}")
        print(f"  Matrice de confusion :")
        print(f"    TN={tn}  FP={fp}")
        print(f"    FN={fn}  TP={tp}")
        print()

    results["meilleur_modele"] = best_model_name
    results["meilleur_auc"] = round(best_score, 3)

    # Refit the best model for final reporting.
    best_model = models[best_model_name]
    best_model.fit(X_scaled, y)
    y_pred_best = best_model.predict(X_scaled)
    try:
        y_proba_best = best_model.predict_proba(X_scaled)[:, 1]
    except Exception:
        df_scores = best_model.decision_function(X_scaled)
        y_proba_best = (df_scores - df_scores.min()) / (df_scores.max() - df_scores.min()) if df_scores.max() > df_scores.min() else np.zeros_like(df_scores)

    cm_best = confusion_matrix(y, y_pred_best)
    tn_b, fp_b, fn_b, tp_b = cm_best.ravel()

    print(f"MEILLEUR MODÈLE : {best_model_name.upper()}")
    print(f"  AUC-ROC : {results['meilleur_auc']}")
    print()

    print("ANALYSE DES ERREURS :")
    print(f"  Faux positifs (prédire 'scientifique' à tort) : {fp_b}")
    print(f"    → Risque pédagogique : orienter un élève littéraire vers scientifique.")
    print(f"  Faux négatifs (prédire 'littéraire' à tort) : {fn_b}")
    print(f"    → Risque pédagogique : freiner un élève scientifique vers littéraire.")
    print()

    print("RECOMMANDATIONS D'UTILISATION :")
    print(f"  - Ce modèle atteint une AUC-ROC de {results['meilleur_auc']}, "
          f"ce qui indique une {'bonne' if best_score > 0.7 else 'acceptable'} capacité prédictive.")
    print(f"  - Il faut considérer cette prédiction comme UNE SUGGESTION, pas une décision définitive.")
    print(f"  - Le conseil de classe doit conserver l'autorité finale sur l'orientation.")
    print(f"  - Un suivi des faux positifs/négatifs aidera à affiner le modèle avec le temps.")
    print()

    # Visualize the best model results.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    cm_best = confusion_matrix(y, y_pred_best)
    im = axes[0, 0].imshow(cm_best, interpolation="nearest", cmap="Blues")
    axes[0, 0].set_xlabel("Prédiction")
    axes[0, 0].set_ylabel("Vraie valeur")
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_yticks([0, 1])
    axes[0, 0].set_xticklabels(["Littéraire", "Scientifique"])
    axes[0, 0].set_yticklabels(["Littéraire", "Scientifique"])
    axes[0, 0].set_title(f"Matrice de confusion ({best_model_name})")
    for i in range(2):
        for j in range(2):
            axes[0, 0].text(j, i, cm_best[i, j], ha="center", va="center", color="white")
    plt.colorbar(im, ax=axes[0, 0])

    fpr, tpr, _ = roc_curve(y, y_proba_best)
    roc_auc = auc(fpr, tpr)
    axes[0, 1].plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {roc_auc:.3f}")
    axes[0, 1].plot([0, 1], [0, 1], "r--", linewidth=1, label="Aléatoire")
    axes[0, 1].set_xlabel("Taux de faux positifs")
    axes[0, 1].set_ylabel("Taux de vrais positifs")
    axes[0, 1].set_title("Courbe ROC")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    cv_scores_dict = {
        name: cross_val_score(model, X_scaled, y, cv=skf, scoring="roc_auc").mean()
        for name, model in models.items()
    }
    axes[1, 0].bar(cv_scores_dict.keys(), cv_scores_dict.values(), color="steelblue")
    axes[1, 0].set_ylabel("AUC-ROC (validation croisée)")
    axes[1, 0].set_title("Comparaison des modèles")
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(alpha=0.3, axis="y")
    for i, (name, score) in enumerate(cv_scores_dict.items()):
        axes[1, 0].text(i, score + 0.02, f"{score:.3f}", ha="center")

    prob_bins = np.linspace(0, 1, 11)
    axes[1, 1].hist(y_proba_best[y == 0], bins=prob_bins, alpha=0.6, label="Littéraire (vrai)", color="blue")
    axes[1, 1].hist(y_proba_best[y == 1], bins=prob_bins, alpha=0.6, label="Scientifique (vrai)", color="red")
    axes[1, 1].set_xlabel("Probabilité prédite (Scientifique)")
    axes[1, 1].set_ylabel("Fréquence")
    axes[1, 1].set_title("Distribution des probabilités")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    output_path = Path(dataset_path).parent.parent / "report" / "q4_classification.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Graphiques sauvegardés : {output_path}")

    results_path = output_path.with_suffix(".json")
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nRÉSUMÉ POUR LA DIRECTION :")
    print("-" * 60)
    print(
        f"Oui, il est possible de suggérer automatiquement une orientation à partir des deux mesures disponibles. "
        f"Le modèle choisi ('{best_model_name}') atteint une AUC-ROC de {results['meilleur_auc']}, "
        f"ce qui indique une capacité prédictive {'bonne' if best_score > 0.7 else 'acceptable'}. "
        f"Cependant, cette prédiction doit rester une AIDE À LA DÉCISION et non une décision automatique. "
        f"Le conseil de classe conserve l'autorité finale. Les risques pédagogiques (mauvaise orientation) "
        f"doivent être gérés en révisant les cas signalés comme douteux."
    )
    plt.close()


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.csv"
    analyze_classification(dataset_path)
