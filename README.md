# INF232_TP_GROUPE68 — Thème D : Établissement scolaire secondaire

## Contexte
Ce dépôt contient le travail pratique du groupe INF232 autour du thème D : établissement scolaire secondaire. Il met en œuvre un générateur de données synthétiques déterministe, ainsi que quatre analyses statistiques complètes :
- Q1 : analyse univariée,
- Q2 : analyse bivariée,
- Q3 : clustering non supervisé,
- Q4 : classification supervisée.

## Structure du dépôt
- data/ : fichiers CSV et métadonnées générés,
- report/ : graphiques et résultats JSON des analyses,
- scripts_analyses/ : scripts Python de génération et d’analyse,
- requirements.txt : dépendances Python nécessaires,
- vscode_settings/ : configuration VS Code.

## Exécution rapide
Pour générer le dataset :
```powershell
python scripts_analyses/generate_data.py --chef "AMBASYLVAIN"
```

Pour exécuter les analyses :
```powershell
python scripts_analyses/analysis_q1.py data/dataset.csv
python scripts_analyses/analysis_q2.py data/dataset.csv
python scripts_analyses/analysis_q3.py data/dataset.csv
python scripts_analyses/analysis_q4.py data/dataset.csv
```

## Résultats obtenus
Le projet génère actuellement un dataset de 250 élèves avec la graine 1474695318. Les résultats sont enregistrés dans le dossier report/ sous forme d’images PNG et de fichiers JSON.

## Notes
Les données sont synthétiques et conçues uniquement pour un usage pédagogique.