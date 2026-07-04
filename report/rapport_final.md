# Rapport final — INF232_TP_GROUPE68

## Thème choisi
Thème D : **Établissement scolaire secondaire**.

## 1. Présentation du projet
Ce projet met en œuvre une application Python capable de générer un jeu de données synthétique reproducible à partir du nom du chef de groupe, puis de répondre aux quatre volets d’analyse du thème D :
- Q1 : analyse univariée,
- Q2 : analyse bivariée,
- Q3 : classification non supervisée,
- Q4 : classification supervisée.

Le code source est fourni dans le dossier `scripts_analyses/` et les données générées sont dans `data/`. Les résultats et visuels se trouvent dans `report/`.

## 2. Mécanisme de génération des données
### 2.1 Objectif
Le générateur doit être déterministe : le même nom de chef de groupe doit produire exactement le même jeu de données.

### 2.2 Algorithme précis de génération
Le mécanisme de génération utilise les étapes suivantes :

1. Normalisation du nom du chef de groupe :
   - Transformation Unicode NFKD pour décomposer les caractères accentués.
   - Encodage en ASCII afin de supprimer les accents.
   - Passage en majuscules.
   - Suppression de tout caractère autre qu’une lettre A–Z.

2. Construction d’une graine numérique :
   - Pour chaque caractère normalisé, calculer `seed = (seed * 31 + ord(char) + index) % 2_147_483_647`.
   - Ajouter un offset fixe de `10_000_003` pour éviter les petites graines triviales.

3. Création d’un générateur pseudo-aléatoire :
   - `rng = random.Random(seed)`.
   - Cela garantit que toutes les valeurs générées sont reproductibles à partir du seul nom du chef.

4. Génération de 250 élèves :
   - Trois bandes de performances sont simulées pour créer une distribution réaliste :
     - 25 % des élèves dans une bande moyenne-basse,
     - 45 % dans une bande moyenne,
     - 30 % dans une bande élevée.
   - `score_eval` est obtenu par une valeur de base dans ces bandes, plus un bruit uniforme `[-8, +8]`, puis borné entre 0 et 100.
   - `temps_etude_heures` est dérivé du score : `4.5 + 0.16 * score_eval + bruit` avec bruit `[-2.2, +2.2]`, borné entre 2 et 25 heures.
   - `assiduite_pct` est dérivée du score : `36 + 0.45 * score_eval + bruit` avec bruit `[-7, +7]`, bornée entre 0 et 100 %.

5. Détermination de l’orientation recommandée :
   - On calcule un signal synthétique `signal = 0.65 * score_eval + 0.25 * assiduite_pct + 0.3 * temps_etude_heures`.
   - Si l’élève est très bon et très assidu (`score_eval >= 82`, `assiduite_pct >= 80`, `temps_etude_heures >= 13`), l’orientation est `scientifique`.
   - Si l’élève est en difficulté (`score_eval <= 46` ou `assiduite_pct <= 34`), l’orientation est `littéraire`.
   - Sinon, on tranche selon `signal > 103`.

### 2.3 Justification des choix
- La normalisation du nom garantit l’unicité et l’absence de sensibilité aux accents, espaces ou tirets.
- Le calcul de la graine par hash rolling est simple, lisible, et offre une bonne dispersion.
- Le modèle de génération de score, temps d’étude et assiduité crée une dépendance cohérente entre variables : plus un élève est performant, plus il travaille et est assidu.
- La décision d’orientation utilise des seuils clairs et un signal additionnel afin de refléter une logique pédagogique plausible.

### 2.4 Graine obtenue pour notre groupe
- Nom du chef de groupe utilisé : **AMBASYLVAIN**
- Graine numérique obtenue : **1474695318**

### 2.5 Extrait représentatif des données produites
| id | score_eval | temps_etude_heures | assiduite_pct | orientation_recommandee |
|----|------------|--------------------|---------------|-------------------------|
| 1  | 51.0       | 13.6               | 63.5          | littéraire              |
| 2  | 70.0       | 15.2               | 61.4          | littéraire              |
| 3  | 56.9       | 13.8               | 67.2          | littéraire              |
| 4  | 66.3       | 16.6               | 72.0          | littéraire              |
| 5  | 93.5       | 19.6               | 71.1          | littéraire              |

Cet extrait permet de vérifier la cohérence entre le nom du chef, la graine et le jeu de données.

## 3. Outils et provenance du code
- Code de génération : `scripts_analyses/generate_data.py`
- Analyses : `scripts_analyses/analysis_q1.py`, `scripts_analyses/analysis_q2.py`, `scripts_analyses/analysis_q3.py`, `scripts_analyses/analysis_q4.py`
- Données générées : `data/dataset.csv`, `data/dataset.meta.json`
- Résultats produits : `report/q1_univariate.*`, `report/q2_bivariate.*`, `report/q3_clustering.*`, `report/q4_classification.*`
- Dépendances Python : `requirements.txt`

## 4. Réponse aux quatre questions du thème
Le traitement complet mobilise les quatre blocs du cours : statistique univariée, bivariée, classification non supervisée et classification supervisée.

### 4.1 Question 1 — Analyse univariée
#### Objectif
Décrire la distribution des scores d’évaluation et détecter les valeurs atypiques.

#### Méthodes utilisées
- Calcul des mesures de tendance centrale : moyenne, médiane.
- Calcul des mesures de dispersion : écart-type, intervalle interquartile.
- Mesure de la forme : skewness, kurtosis.
- Détection d’outliers par règle IQR et par z-score.
- Visualisation : histogramme et boîte à moustaches.

#### Résultats
- Effectif : **250 élèves**
- Moyenne : **69.17 / 100**
- Médiane : **69.55 / 100**
- Écart-type : **16.12**
- Min : **37.7**, Max : **100.0**
- Q1 : **56.75**, Q3 : **81.88**, IQR = **25.12**
- Skewness : **-0.03** (distribution presque symétrique)
- Kurtosis : **-1.04** (queue légèrement plus fine qu’une normale)
- Aucune valeur atypique détectée par la règle IQR.

#### Interprétation
La distribution des scores est équilibrée et centrée autour de 70 points. Les élèves sont bien répartis entre les trois niveaux définis par le générateur, sans cas extrêmes significatifs.

### 4.2 Question 2 — Analyse bivariée
#### Objectif
Analyser la relation entre le temps d’étude hebdomadaire et le score d’évaluation.

#### Méthodes utilisées
- Corrélations : Pearson et Spearman.
- Ajustement d’une droite de régression linéaire.
- Évaluation de l’ajustement : coefficient de détermination R², RMSE.
- Vérification de la significativité statistique (p-valeur).
- Visualisation : nuage de points, droite de régression, diagnostic des résidus.

#### Résultats
- Corrélation de Pearson : **0.895**
- Corrélation de Spearman : **0.900**
- Coefficient de détermination : **R² = 0.802**
- Régression : `score = -11.0 + 5.1671 × temps_etude`
- RMSE : **7.17**
- Écart-type des résidus : **7.18**
- P-valeur : **0.0** (relation très significative)

#### Interprétation
Il existe une relation forte et positive entre le temps d’étude et le score. En moyenne, une heure supplémentaire d’étude est associée à une augmentation d’environ 5 points. Cette relation explique 80.2 % de la variance des scores, ce qui est élevé et montre que le temps d’étude est un bon prédicteur dans ce jeu de données.

### 4.3 Question 3 — Classification non supervisée
#### Objectif
Identifier des profils naturels d’élèves sans utiliser l’orientation recommandée comme label.

#### Méthodes utilisées
- Standardisation des variables pour rendre score, temps d’étude et assiduité comparables.
- K-means clustering.
- Sélection du nombre de clusters basé sur le score de silhouette.
- Profilage des clusters en fonction des moyennes des variables.

#### Résultats
- Nombre optimal de clusters : **2**
- Score de silhouette : **0.502**

Profils identifiés :
- **Profil 1** : 122 élèves (48.8 %)
  - Score moyen : **55.5**
  - Temps d’étude moyen : **13.4 h/semaine**
  - Assiduité moyenne : **60.2 %**
  - Description : élèves moyens, implication variable.
- **Profil 2** : 128 élèves (51.2 %)
  - Score moyen : **82.2**
  - Temps d’étude moyen : **17.6 h/semaine**
  - Assiduité moyenne : **74.2 %**
  - Description : élèves performants et studieux.

#### Interprétation
Le clustering montre deux grands groupes cohérents : un groupe plus faible et un groupe plus performant. Ces profils peuvent guider des actions pédagogiques différenciées : soutien et remédiation pour le premier groupe, approfondissement ou valorisation pour le second.

### 4.4 Question 4 — Classification supervisée
#### Objectif
Prédire l’orientation recommandée (`scientifique` / `littéraire`) à partir des mesures disponibles.

#### Méthodes utilisées
- Encodage de la variable cible en binaire.
- Standardisation des caractéristiques.
- Validation croisée stratifiée 5 plis.
- Modèles comparés : régression logistique, arbre de décision, forêt aléatoire.
- Mesures d’évaluation : AUC-ROC, accuracy, précision, rappel, F1, matrice de confusion.

#### Résultats
- Meilleur modèle : **Random Forest**
- AUC-ROC : **1.0**
- Accuracy : **1.0**
- Precision : **1.0**
- Recall : **1.0**
- F1-score : **1.0**
- Matrice de confusion du meilleur modèle :
  - TN = 228, FP = 0
  - FN = 0, TP = 22

#### Interprétation
Le modèle random forest prédit parfaitement l’orientation recommandée sur ce jeu de données synthétique. Cela reflète à la fois la cohérence interne du générateur et le fait que l’orientation est déterminée par une fonction connue des variables disponibles.

## 5. Discussion générale
### 5.1 Validité pédagogique
Le jeu de données est synthétique, mais construit de manière à refléter des corrélations plausibles entre la note, le temps d’étude et l’assiduité. Les choix de génération permettent de simuler des profils réalistes et d’appliquer des méthodes vues en cours.

### 5.2 Limites
- Les données sont artificielles et ne doivent pas être interprétées comme des observations réelles.
- La classification supervisée est très performante ici car l’orientation est générée à partir d’une logique interne directement liée aux variables d’entrée.
- Un cas réel pourrait nécessiter des variables supplémentaires et des modèles plus robustes.

## 6. Conclusion
Le thème D a été traité de manière complète :
- une génération déterministe et justifiée des données,
- une analyse univariée précise,
- une étude bivariée rigoureuse,
- une découverte de profils par clustering non supervisé,
- une prédiction d’orientation par classification supervisée.

L’ensemble du travail est disponible dans le dépôt et peut être exécuté avec les scripts Python fournis.

## Annexes
### Annexe 1 : Extrait du code de génération
```python
def normalize_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Z]", "", ascii_text.upper())


def build_seed(chef_name: str) -> int:
    normalized = normalize_name(chef_name)
    seed = 0
    for index, char in enumerate(normalized):
        seed = (seed * 31 + ord(char) + index) % 2_147_483_647
    return seed + 10_000_003


def generate_dataset(chef_name: str, count: int = 250) -> Tuple[List[Dict[str, object]], int]:
    seed = build_seed(chef_name)
    rng = random.Random(seed)
    rows = []
    for student_id in range(1, count + 1):
        profile = rng.random()
        if profile < 0.25:
            base_score = rng.uniform(42, 58)
        elif profile < 0.70:
            base_score = rng.uniform(58, 78)
        else:
            base_score = rng.uniform(78, 95)

        score_eval = round(clip(base_score + rng.uniform(-8, 8), 0, 100), 1)
        temps_etude = round(clip(4.5 + 0.16 * score_eval + rng.uniform(-2.2, 2.2), 2.0, 25.0), 1)
        assiduite = round(clip(36 + 0.45 * score_eval + rng.uniform(-7.0, 7.0), 0.0, 100.0), 1)

        signal = score_eval * 0.65 + assiduite * 0.25 + temps_etude * 0.3
        if score_eval >= 82 and assiduite >= 80 and temps_etude >= 13:
            orientation = "scientifique"
        elif score_eval <= 46 or assiduite <= 34:
            orientation = "littéraire"
        else:
            orientation = "scientifique" if signal > 103 else "littéraire"

        rows.append({
            "id": student_id,
            "score_eval": score_eval,
            "temps_etude_heures": temps_etude,
            "assiduite_pct": assiduite,
            "orientation_recommandee": orientation,
        })

    return rows, seed
```

### Annexe 2 : Extrait de données représentatif
| id | score_eval | temps_etude_heures | assiduite_pct | orientation_recommandee |
|----|------------|--------------------|---------------|-------------------------|
| 1  | 51.0       | 13.6               | 63.5          | littéraire              |
| 2  | 70.0       | 15.2               | 61.4          | littéraire              |
| 3  | 56.9       | 13.8               | 67.2          | littéraire              |
| 4  | 66.3       | 16.6               | 72.0          | littéraire              |
| 5  | 93.5       | 19.6               | 71.1          | littéraire              |
