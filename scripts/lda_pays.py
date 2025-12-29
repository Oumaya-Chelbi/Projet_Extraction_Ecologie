# =====================================================
# SCORE ÉCOLOGIQUE PAR PAYS (AVEC LDA ÉCOLOGIQUE)
# =====================================================

import os
import re
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
import nltk
from nltk.corpus import stopwords

# PARAMÈTRES

CHEMIN_CORPUS_GLOBAL = "data/processed/EUPDCorp_TXT"   # corpus normal, non filtré
MODELE_LDA_ECO_PATH = "models/lda_ecologie.model"
DICTIONNAIRE_ECO_PATH = "models/lda_ecologie.dict"

DOSSIER_SORTIE = "results/lda_pays_ecologie"

MIN_TOKENS = 20      # longueur min d'un discours pour être pris en compte
TOPICS_ECO = list(range(10))   # topics 0–9 du modèle écologique [file:98]

# STOPWORDS & PREPROCESSING (doivent être cohérents avec le modèle LDA écolo)

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stop_words.update([
    "would", "could", "also", "may", "must", "one", "said",
    "like", "can", "us", "mr", "madam",
    "parliament", "commission", "european", "union"
])

def nettoyer_texte(texte):
    texte = texte.lower()
    texte = re.sub(r"[^a-z\s]", " ", texte)
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()

def tokenizer(texte):
    return [
        mot for mot in texte.split()
        if len(mot) > 3 and mot not in stop_words
    ]

# CHARGEMENT DU MODÈLE LDA ÉCOLOGIQUE

os.makedirs(DOSSIER_SORTIE, exist_ok=True)

dictionary = corpora.Dictionary.load(DICTIONNAIRE_ECO_PATH)
lda_eco = LdaModel.load(MODELE_LDA_ECO_PATH)

# SCORE ÉCOLOGIQUE PAR DISCOURS (CORPUS GLOBAL)

doc_rows = []

for dossier_annee in sorted(os.listdir(CHEMIN_CORPUS_GLOBAL)):
    chemin_annee = os.path.join(CHEMIN_CORPUS_GLOBAL, dossier_annee)
    if not os.path.isdir(chemin_annee):
        continue

    for fichier in os.listdir(chemin_annee):
        if not fichier.endswith(".txt"):
            continue

        chemin_fichier = os.path.join(chemin_annee, fichier)
        pays = os.path.splitext(fichier)[0]  # FRA.txt -> FRA

        with open(chemin_fichier, encoding="utf-8", errors="ignore") as f:
            texte = f.read()

        texte = nettoyer_texte(texte)
        tokens = tokenizer(texte)

        if len(tokens) < MIN_TOKENS:
            continue

        # projeter le document dans l'espace du modèle LDA écolo
        bow = dictionary.doc2bow(tokens)
        if len(bow) == 0:
            continue

        # distribution de topics
        doc_topics = lda_eco.get_document_topics(bow, minimum_probability=0.0)

        # score écologique = somme des probabilités des topics écolo
        score_eco = sum(prob for tid, prob in doc_topics if tid in TOPICS_ECO)

        doc_rows.append({
            "annee": dossier_annee,
            "pays": pays,
            "fichier": chemin_fichier,
            "score_eco": score_eco
        })

df_docs = pd.DataFrame(doc_rows)
docs_scores_path = os.path.join(DOSSIER_SORTIE, "scores_ecologiques_par_discours_global.csv")
df_docs.to_csv(docs_scores_path, index=False)
print(f" Scores écologiques par discours (corpus global) : {docs_scores_path}")

# AGRÉGATION PAR PAYS : TOP 10

# score total par pays (volume d'écologie)
df_pays = (
    df_docs
    .groupby("pays", as_index=False)
    ["score_eco"]
    .sum()
    .rename(columns={"score_eco": "score_eco_total"})
)

# trier du pays le plus au moins écologique
df_pays_sorted = df_pays.sort_values("score_eco_total", ascending=False)

pays_scores_path = os.path.join(DOSSIER_SORTIE, "classement_pays_ecologie_global.csv")
df_pays_sorted.to_csv(pays_scores_path, index=False)

print(" Top 10 pays qui ont le plus parlé d'écologie (score total):")
print(df_pays_sorted.head(10))

# POUR CHAQUE TOP PAYS : ANNÉE LA PLUS ÉCOLOGIQUE

top10_pays = df_pays_sorted.head(10)["pays"].tolist()

rows_top10 = []

for pays in top10_pays:
    df_p = df_docs[df_docs["pays"] == pays]

    # score par année pour ce pays
    df_p_annee = (
        df_p
        .groupby("annee", as_index=False)
        ["score_eco"]
        .sum()
        .rename(columns={"score_eco": "score_eco_annee"})
    )

    # année avec score max
    best_row = df_p_annee.sort_values("score_eco_annee", ascending=False).iloc[0]

    rows_top10.append({
        "pays": pays,
        "annee_max_eco": best_row["annee"],
        "score_eco_max_annee": best_row["score_eco_annee"],
        "score_eco_total_pays": float(df_p_annee["score_eco_annee"].sum())
    })

df_top10 = pd.DataFrame(rows_top10)

top10_path = os.path.join(DOSSIER_SORTIE, "top10_pays_et_annee_max_eco.csv")
df_top10.to_csv(top10_path, index=False)

print(" Top 10 pays + année où ils ont le plus parlé d'écologie :")
print(df_top10)

print(f"Résumé top 10 : {top10_path}")
