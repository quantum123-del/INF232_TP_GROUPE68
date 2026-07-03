#!/usr/bin/env python3
"""Question 4: Simplified classification without matplotlib."""

import sys
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score, 
                            roc_auc_score)

def analyze_q4_simplified(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    X = df[['score_eval', 'temps_etude_heures', 'assiduite_pct']]
    y = df['orientation_recommandee']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print("\n" + "="*70)
    print("QUESTION 4 : CLASSIFICATION SUPERVISÉE - PRÉDICTION D'ORIENTATION")
    print("="*70)
    
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    }
    
    results = {}
    best_f1 = 0
    best_model_name = None
    
    for model_name, model in models.items():
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        results[model_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc_roc": float(auc),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "error_rate": float((fp + fn) / len(y_test) * 100)
        }
        
        print(f"\n{model_name}:")
        print(f"  CV Score : {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        print(f"  F1-score : {f1:.3f}")
        print(f"  AUC-ROC : {auc:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = model_name
    
    print(f"\n--- MEILLEUR MODÈLE : {best_model_name} ---")
    best_results = results[best_model_name]
    print(f"Taux d'erreur : {best_results['error_rate']:.1f}%")
    print(f"Fiabilité : {'BONNE' if best_f1 > 0.7 else 'MODÉRÉE' if best_f1 > 0.6 else 'FAIBLE'}")
    
    with open('report/q4_results.json', 'w') as f:
        json.dump({
            "best_model": best_model_name,
            "all_models": results,
            "recommendation": "À valider avec le conseil de classe"
        }, f, indent=2)
    
    print(f"\n✓ Résultats sauvegardés : report/q4_results.json")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.csv"
    analyze_q4_simplified(csv_file)
