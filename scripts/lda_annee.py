# =====================================================
# LDA PAR ANNÉE – ÉVOLUTION DU DISCOURS ÉCOLOGIQUE
# =====================================================

import os
import re
from gensim import corpora
from gensim.models import LdaModel
import nltk
from nltk.corpus import stopwords
import pandas as pd

# PARAMÈTRES

CHEMIN_CORPUS = "data/processed/EUPDCorp_TXT_Ecologie1"
DOSSIER_SORTIE = "results/lda_ecologie_par_annee"

NUM_TOPICS = 6
PASSES = 10
TOP_MOTS = 10
RANDOM_STATE = 42
MIN_TOKENS = 20

# MOTS-CLÉS ÉCOLOGIQUES

MOTS_CLES_ECO = [
    "environment", "environmental", "climate", "global", "warming",
    "biodiversity", "ecosystem", "pollution", "sustainable",
    "sustainability", "renewable", "energy", "carbon",
    "greenhouse", "emissions", "ecological", "atmospheric",
    "ozone", "waste"
]

# STOPWORDS

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stop_words.update([
    "would", "could", "also", "may", "must", "one", "said",
    "like", "can", "us", "mr", "madam",
    "parliament", "commission", "european", "union"
])

# FONCTIONS TEXTE

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

# LDA PAR ANNÉE

os.makedirs(DOSSIER_SORTIE, exist_ok=True)

resume_rows = []

for dossier_annee in sorted(os.listdir(CHEMIN_CORPUS)):
    chemin_annee = os.path.join(CHEMIN_CORPUS, dossier_annee)
    if not os.path.isdir(chemin_annee):
        continue

    print(f"LDA pour l'année {dossier_annee}")

    documents = []
    tokens_par_doc = []

    for fichier in os.listdir(chemin_annee):
        if not fichier.endswith(".txt"):
            continue

        with open(os.path.join(chemin_annee, fichier),
                  encoding="utf-8", errors="ignore") as f:
            texte = nettoyer_texte(f.read())
            tokens = tokenizer(texte)

            if len(tokens) >= MIN_TOKENS:
                documents.append(tokens)
                tokens_par_doc.append(tokens)

    print(f"Documents retenus : {len(documents)}")

    if len(documents) < 5:
        print("⚠️ Trop peu de documents, année ignorée")
        continue

    # -------- LDA --------
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        passes=PASSES,
        random_state=RANDOM_STATE
    )

    # -------- Sortie par année --------
    dossier_out_annee = os.path.join(DOSSIER_SORTIE, dossier_annee)
    os.makedirs(dossier_out_annee, exist_ok=True)

    rows_topics = []

    for i, topic in lda.print_topics(num_words=TOP_MOTS):
        for mot, poids in lda.show_topic(i, TOP_MOTS):
            rows_topics.append({
                "annee": dossier_annee,
                "topic": i,
                "mot": mot,
                "poids": poids
            })

    pd.DataFrame(rows_topics).to_csv(
        os.path.join(dossier_out_annee, f"lda_{dossier_annee}.csv"),
        index=False
    )

    # -------- POIDS DES TOPICS --------
    topic_weights = {i: 0 for i in range(NUM_TOPICS)}

    for doc in corpus:
        for topic_id, weight in lda.get_document_topics(doc):
            topic_weights[topic_id] += weight

    nb_topics_actifs = sum(1 for v in topic_weights.values() if v > 1)

    for topic_id, poids_total in topic_weights.items():
        resume_rows.append({
            "annee": dossier_annee,
            "type": "topic_weight",
            "topic": topic_id,
            "valeur": poids_total
        })

    resume_rows.append({
        "annee": dossier_annee,
        "type": "nb_topics_actifs",
        "topic": None,
        "valeur": nb_topics_actifs
    })

    # -------- FRÉQUENCE DES MOTS ÉCOLOGIQUES --------
    freq_eco = {mot: 0 for mot in MOTS_CLES_ECO}

    for tokens in tokens_par_doc:
        for mot in tokens:
            if mot in freq_eco:
                freq_eco[mot] += 1

    for mot, freq in freq_eco.items():
        resume_rows.append({
            "annee": dossier_annee,
            "type": "mot_cle_eco",
            "topic": mot,
            "valeur": freq
        })

# SAUVEGARDE GLOBALE

df_resume = pd.DataFrame(resume_rows)
df_resume.to_csv(
    os.path.join(DOSSIER_SORTIE, "resume_comparatif_annees.csv"),
    index=False
)

print(" LDA par année + comparaison temporelle terminés")
