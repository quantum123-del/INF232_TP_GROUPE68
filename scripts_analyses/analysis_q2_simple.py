#!/usr/bin/env python3
"""Question 2: Simplified bivariate analysis without matplotlib."""

import sys
import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def analyze_q2_simplified(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    x = df['assiduite_pct'].values.reshape(-1, 1)
    y = df['score_eval'].values
    
    # Correlation
    correlation = np.corrcoef(y, x.flatten())[0, 1]
    
    # Linear regression
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = np.mean(np.abs(y - y_pred))
    
    # Residuals
    residuals = y - y_pred
    std_error = np.std(residuals)
    ci_95 = 1.96 * std_error
    
    results = {
        "correlation": float(correlation),
        "r2_score": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "intercept": float(model.intercept_),
        "slope": float(model.coef_[0]),
        "std_residuals": float(std_error),
        "ci_95": float(ci_95),
        "reliability": "Fiable" if r2 > 0.6 else "Modérée" if r2 > 0.4 else "Non fiable"
    }
    
    print("\n" + "="*70)
    print("QUESTION 2 : ANALYSE BIVARIÉE ET CAPACITÉ PRÉDICTIVE")
    print("="*70)
    print(f"\nCorrélation de Pearson : {correlation:.4f}")
    print(f"R² (variance expliquée) : {r2:.4f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"Équation : score = {model.intercept_:.2f} + {model.coef_[0]:.4f} × assiduité")
    print(f"Fiabilité : {results['reliability']}")
    print(f"IC 95% : ±{ci_95:.2f}")
    
    # Save results
    with open('report/q2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Résultats sauvegardés : report/q2_results.json")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.csv"
    analyze_q2_simplified(csv_file)
