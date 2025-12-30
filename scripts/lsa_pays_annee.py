# SCORE ÉCOLOGIQUE LSA PAR PAYS + ANNÉE MAX

import os
import re
import nltk
from nltk.corpus import stopwords
import joblib
import numpy as np
import pandas as pd

# PARAMÈTRES

# Corpus global (non filtré)
CHEMIN_CORPUS_GLOBAL = "data/processed/EUPDCorp_TXT"

# Dossier contenant les modèles LSA et TF-IDF
MODELS_DIR = "models_lsa"

# Dossier où on va sauvegarder les scores par pays
DOSSIER_SORTIE = "results/lsa_pays_ecologie"

# Nombre de premières composantes LSA à utiliser pour le score écologique
N_COMPONENTS_ECO_USED = 5

# On crée le dossier de sortie si nécessaire
os.makedirs(DOSSIER_SORTIE, exist_ok=True)

# STOPWORDS

# On télécharge les stopwords et on ajoute des mots fréquents mais peu informatifs
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stop_words.update([
    "would", "could", "also", "may", "must", "one", "said",
    "like", "can", "us", "mr", "madam",
    "parliament", "commission", "european", "union"
])

# FONCTIONS TEXTE

# Nettoyage simple du texte
def nettoyer_texte(texte):
    texte = texte.lower()
    texte = re.sub(r"[^a-z\s]", " ", texte)
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()

# CHARGEMENT DES MODÈLES LSA

# On charge le vectorizer TF-IDF et le modèle LSA précédemment appris
vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_eco.pkl"))
svd = joblib.load(os.path.join(MODELS_DIR, "lsa_eco.pkl"))

# SCORE ÉCOLOGIQUE PAR DISCOURS (CORPUS GLOBAL)

doc_rows = []  # On stocke ici les scores par discours

# On parcourt le corpus année par année
for dossier_annee in sorted(os.listdir(CHEMIN_CORPUS_GLOBAL)):
    chemin_annee = os.path.join(CHEMIN_CORPUS_GLOBAL, dossier_annee)
    if not os.path.isdir(chemin_annee):
        continue

    for fichier in os.listdir(chemin_annee):
        if not fichier.endswith(".txt"):
            continue

        chemin_fichier = os.path.join(chemin_annee, fichier)
        pays = os.path.splitext(fichier)[0]  # FRA.txt -> FRA

        # Lecture et nettoyage du texte
        with open(chemin_fichier, encoding="utf-8", errors="ignore") as f:
            texte = f.read()
        texte = nettoyer_texte(texte)

        # TF-IDF avec le vectorizer du modèle écologique
        X_doc = vectorizer.transform([texte])
        if X_doc.nnz == 0:
            continue  # On ignore les documents vides après vectorisation

        # Projection LSA
        x_lsa = svd.transform(X_doc)[0]  # vecteur 1D

        # Score écologique = norme des premières composantes
        k = min(N_COMPONENTS_ECO_USED, len(x_lsa))
        score_eco = float(np.linalg.norm(x_lsa[:k]))

        # On stocke le résultat pour ce discours
        doc_rows.append({
            "annee": dossier_annee,
            "pays": pays,
            "fichier": chemin_fichier,
            "score_eco": score_eco
        })

# On met les résultats dans un DataFrame
df_docs = pd.DataFrame(doc_rows)
docs_scores_path = os.path.join(DOSSIER_SORTIE, "scores_ecologiques_lsa_par_discours.csv")
df_docs.to_csv(docs_scores_path, index=False)
print(f" Scores LSA écologiques par discours : {docs_scores_path}")

# AGRÉGATION PAR PAYS (TOP 10) + ANNÉE MAX

# On calcule le score total par pays
df_pays = (
    df_docs
    .groupby("pays", as_index=False)
    ["score_eco"]
    .sum()
    .rename(columns={"score_eco": "score_eco_total"})
)

# On trie les pays du plus au moins écologique
df_pays_sorted = df_pays.sort_values("score_eco_total", ascending=False)
pays_scores_path = os.path.join(DOSSIER_SORTIE, "classement_pays_lsa_ecologie.csv")
df_pays_sorted.to_csv(pays_scores_path, index=False)

print("Top 10 pays (LSA, score total éco) :")
print(df_pays_sorted.head(10))

# On récupère les 10 pays les plus écologiques
top10_pays = df_pays_sorted.head(10)["pays"].tolist()
rows_top10 = []

for pays in top10_pays:
    df_p = df_docs[df_docs["pays"] == pays]

    # Score par année pour ce pays
    df_p_annee = (
        df_p
        .groupby("annee", as_index=False)
        ["score_eco"]
        .sum()
        .rename(columns={"score_eco": "score_eco_annee"})
    )

    # Année avec le score maximal
    best_row = df_p_annee.sort_values("score_eco_annee", ascending=False).iloc[0]

    rows_top10.append({
        "pays": pays,
        "annee_max_eco": best_row["annee"],
        "score_eco_max_annee": best_row["score_eco_annee"],
        "score_eco_total_pays": float(df_p_annee["score_eco_annee"].sum())
    })

# On compile les résultats top 10
df_top10 = pd.DataFrame(rows_top10)
top10_path = os.path.join(DOSSIER_SORTIE, "top10_pays_et_annee_max_lsa_ecologie.csv")
df_top10.to_csv(top10_path, index=False)

print(" Top 10 pays + année la plus écologique (LSA) :")
print(df_top10)
print(f" Résumé top 10 : {top10_path}")
