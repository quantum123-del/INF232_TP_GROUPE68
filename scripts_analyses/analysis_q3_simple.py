#!/usr/bin/env python3
"""Question 3: Simplified clustering without matplotlib."""

import sys
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def analyze_q3_simplified(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    features = df[['score_eval', 'temps_etude_heures', 'assiduite_pct']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Find optimal k
    silhouette_scores = []
    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        sil_score = silhouette_score(features_scaled, labels)
        silhouette_scores.append(sil_score)
    
    optimal_k = 2 + np.argmax(silhouette_scores)
    
    # Final clustering
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = kmeans_final.fit_predict(features_scaled)
    
    print("\n" + "="*70)
    print("QUESTION 3 : CLASSIFICATION NON SUPERVISÉE - PROFILS D'ÉLÈVES")
    print("="*70)
    print(f"\nNombre optimal de clusters : {optimal_k}")
    
    clusters_info = {}
    for cluster_id in range(optimal_k):
        cluster_data = df[df['cluster'] == cluster_id]
        size = len(cluster_data)
        score_mean = cluster_data['score_eval'].mean()
        temps_mean = cluster_data['temps_etude_heures'].mean()
        assid_mean = cluster_data['assiduite_pct'].mean()
        sci_count = (cluster_data['orientation_recommandee'] == 'scientifique').sum()
        lit_count = (cluster_data['orientation_recommandee'] == 'littéraire').sum()
        
        clusters_info[f"profil_{cluster_id+1}"] = {
            "size": int(size),
            "percentage": float(size/len(df)*100),
            "score_mean": float(score_mean),
            "temps_etude_mean": float(temps_mean),
            "assiduite_mean": float(assid_mean),
            "scientifique_count": int(sci_count),
            "litteraire_count": int(lit_count)
        }
        
        print(f"\nProfil {cluster_id + 1} : {size} élèves ({size/len(df)*100:.1f}%)")
        print(f"  Score moyen : {score_mean:.1f}/100")
        print(f"  Temps d'étude : {temps_mean:.1f}h/semaine")
        print(f"  Assiduité : {assid_mean:.1f}%")
    
    # Save results and dataset with clusters
    with open('report/q3_results.json', 'w') as f:
        json.dump(clusters_info, f, indent=2)
    
    df[['id', 'score_eval', 'temps_etude_heures', 'assiduite_pct', 'cluster']].to_csv(
        'data/dataset_with_clusters.csv', index=False
    )
    
    print(f"\n✓ Résultats sauvegardés : report/q3_results.json")
    print(f"✓ Dataset avec clusters : data/dataset_with_clusters.csv")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.csv"
    analyze_q3_simplified(csv_file)
