# =====================================================
# LDA GLOBALE - EUPDCorp (TXT par année / pays)
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
DOSSIER_SORTIE = "results/lda_ecologie"

NUM_TOPICS = 10
PASSES = 10
TOP_MOTS = 12
RANDOM_STATE = 42

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

# CHARGEMENT DU CORPUS

print(" Chargement du corpus EUPDCorp...")

documents = []

for dossier_annee in sorted(os.listdir(CHEMIN_CORPUS)):
    chemin_annee = os.path.join(CHEMIN_CORPUS, dossier_annee)
    if not os.path.isdir(chemin_annee):
        continue

    for fichier in os.listdir(chemin_annee):
        if fichier.endswith(".txt"):
            chemin_fichier = os.path.join(chemin_annee, fichier)
            with open(chemin_fichier, encoding="utf-8", errors="ignore") as f:
                texte = f.read()

            texte = nettoyer_texte(texte)
            tokens = tokenizer(texte)

            if len(tokens) > 20:
                documents.append(tokens)

print(f" Documents analysés : {len(documents)}")

# LDA GLOBALE

print(" Lancement LDA globale...")

dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(doc) for doc in documents]

lda = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    passes=PASSES,
    random_state=RANDOM_STATE
)

# SAUVEGARDE DES RÉSULTATS

os.makedirs(DOSSIER_SORTIE, exist_ok=True)

txt_path = f"{DOSSIER_SORTIE}/lda_global_eupdcorp.txt"
csv_path = f"{DOSSIER_SORTIE}/lda_global_eupdcorp.csv"

rows = []

with open(txt_path, "w", encoding="utf-8") as f:
    for i, topic in lda.print_topics(num_words=TOP_MOTS):
        ligne = f"Topic {i}: {topic}"
        print(ligne)
        f.write(ligne + "\n")

        for mot, poids in lda.show_topic(i, TOP_MOTS):
            rows.append({
                "topic": i,
                "mot": mot,
                "poids": poids
            })

pd.DataFrame(rows).to_csv(csv_path, index=False)

print(" LDA globale terminée")
print(f"TXT : {txt_path}")
print(f" CSV : {csv_path}")

#  création de model et dictionary
os.makedirs("models", exist_ok=True)
dictionary.save("models/lda_ecologie.dict")
lda.save("models/lda_ecologie.model")
