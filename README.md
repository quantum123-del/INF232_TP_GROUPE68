# INF232_TP_GROUPE68 — Thème D : Établissement scolaire secondaire

## 1. Contexte
Ce dépôt contient le travail pratique du groupe INF232 (Thème D : Établissement scolaire secondaire). L'objectif est d'analyser un jeu de données généré de manière déterministe à partir du nom complet du chef de groupe, et de produire :
- une application (code + données) réutilisable, et
- un rapport clair et pédagogique répondant aux quatre questions du thème.

## 2. Objectifs du projet
- Fournir une méthode reproductible de génération de données basées sur la graine dérivée du nom du chef de groupe.
- Réaliser les analyses statistiques demandées :
  - Q1 : Analyse univariée (description, dispersion, valeurs atypiques).
  - Q2 : Analyse bivariée (relation entre deux mesures et possibilités d'estimation).
  - Q3 : Classification non supervisée (découverte de profils d'élèves).
  - Q4 : Classification supervisée (prévision de l'orientation recommandée).
- Produire un rapport interprétable par des non-statisticiens et un code bien organisé et documenté.

## 3. Production attendue
- Un dossier `Application/` contenant le code source (génération, analyses, visualisations).
- Un dossier `data/` contenant les jeux de données générés (.csv) et la graine utilisée.
- Un dossier `report/` avec le rapport final (PDF/MD) et les figures clefs.
- Un court `README.md` (celui-ci) expliquant la feuille de route et les instructions d'exécution.

## 4. Génération des données (exigences)
- Le dataset doit être produit par une fonction déterministe : transformer le nom complet du chef de groupe (prénom + nom, en MAJUSCULES, sans espaces ni accents) en une graine entière.
- Exemple de transformation (expliquer dans le rapport) : concaténation puis somme de codes caractères, ou hash simple suivi d'un modulo.
- Le script `src/generate.py` (ou équivalent) doit accepter le paramètre `--chef "NOMPRENOM"` et produire :
  - un fichier `data/dataset.csv` avec colonnes recommandées : `id`, `score_eval` (0–100), `mesure_secondaire` (ex. heures d'étude / assiduité), `orientation_recommandee` ("scientifique" / "littéraire").
- Taille recommandée : 200–500 élèves (à ajuster et justifier dans le rapport).
- Les bornes et distributions choisies doivent être plausibles et justifiées dans le rapport.

## 5. Méthodologie par question (feuille de route d'analyse)
- Question 1 (Univariée)
  - Calculer : moyenne, médiane, écart-type, quantiles (25%, 75%), skewness.
  - Visualiser : histogramme, boxplot.
  - Détecter valeurs atypiques : règle IQR et z-score.
  - Rédiger un résumé simple pour le proviseur (2–3 phrases).

- Question 2 (Bivariée)
  - Visualiser : nuage de points, nuage avec régression linéaire.
  - Mesurer : coefficient de corrélation (Pearson/Spearman), ajustement linéaire, RMSE.
  - Évaluer la fiabilité d'une estimation depuis la seconde mesure : intervalles de confiance et limites d'application.

- Question 3 (Non supervisée)
  - Prétraitement : standardisation des variables.
  - Méthodes : K-means (avec évaluation par silhouette), clustering hiérarchique en complément.
  - Décrire chaque profil avec des termes compréhensibles (p. ex. "fort en éval, faible assiduité").

- Question 4 (Supervisée)
  - Préparer un jeu d'entraînement/test (cross-validation stratifiée).
  - Algorithmes proposés : régression logistique, arbre de décision, forêt aléatoire.
  - Mesures d'évaluation : matrice de confusion, précision, rappel, F1, AUC-ROC.
  - Discussion des conséquences pratiques d'erreurs (faux positifs / faux négatifs) pour l'élève.

## 6. Contraintes et reproductibilité
- Le générateur doit être déterministe : relancer le script avec le même nom du chef doit produire exactement les mêmes données.
- Le rapport doit inclure : description algorithmique du générateur, la graine numérique obtenue, et un extrait des données pour vérification.
- Aucune donnée réelle ne doit être utilisée sans anonymisation ; ce TP exige des données synthétiques.

## 7. Structure recommandée du dépôt
- `INF232_TP_GROUPE68/`
  - `data/` — jeux de données générés (.csv)
  - `src/` — scripts : `generate.py`, `analysis_q1.py`, `analysis_q2.py`, `clustering.py`, `classification.py`
  - `notebooks/` — notebooks d'exploration (optionnels)
  - `report/` — rapport final et figures
  - `README.md` — feuille de route (ce fichier)

## 8. Instructions d'exécution (exemples)
Pour générer les données :

```powershell
python src/generate.py --chef "NOMPRENOM"
```

Pour lancer les analyses :

```powershell
python src/analysis_q1.py data/dataset.csv
python src/analysis_q2.py data/dataset.csv
python src/clustering.py data/dataset.csv
python src/classification.py data/dataset.csv
```

(Adapter selon le langage choisi ; documenter les dépendances dans `requirements.txt` ou `package.json`.)

## 9. Timeline et jalons (suggestion)
- Jours 1–2 : conception du générateur et choix des variables.
- Jours 3–4 : génération des données et analyses Q1–Q2.
- Jours 5–6 : clustering et classification, évaluation.
- Jour 7 : rédaction du rapport et préparation des graphiques.

## 10. Répartition des tâches (proposition)
- Générateur & validation : Membre A
- Analyse Q1 & Q2 : Membres B et C
- Clustering (Q3) : Membres D et E
- Classification (Q4) : Membres F et G
- Rapport & mise en forme : Membres H, I, J

## 11. Critères d'évaluation (à inclure dans le rapport)
- Reproductibilité du générateur (graine + code).
- Pertinence et justification des choix de variables et distributions.
- Qualité des analyses (méthode, visualisations, interprétation).
- Pertinence des conclusions et discussion des limites.
- Clarté et professionnalisme du rapport et de l'application.

## 12. Notes éthiques
- Indiquer explicitement que les données sont synthétiques et conçues pour un TP.
- Discuter les risques liés à l'automatisation des orientations scolaires et les limites pratiques.

---

Si vous validez ce README, je peux :
- créer le squelette du dépôt (`src/`, `data/`, `report/`) et un script de génération de données minimal,
- ou rédiger directement les scripts d'analyse pour Q1–Q4.


