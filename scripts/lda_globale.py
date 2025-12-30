# LDA GLOBALE -> EUPDCorp (TXT par année / pays) pour Première execution
# Deuxiéme sur le corpus ecologie : EUPDCorp_TXT_Ecologie1

import os
import re
from gensim import corpora
from gensim.models import LdaModel
import nltk
from nltk.corpus import stopwords
import pandas as pd

# PARAMÈTRES

# Chemin vers notre corpus filtré sur l’écologie
CHEMIN_CORPUS = "data/processed/EUPDCorp_TXT_Ecologie1"

# Dossier où on va sauvegarder les résultats globaux
DOSSIER_SORTIE = "results/lda_ecologie"

# Paramètres LDA : nombre de topics, passes, mots par topic
NUM_TOPICS = 10
PASSES = 10
TOP_MOTS = 12
RANDOM_STATE = 42

# STOPWORDS

# On télécharge les stopwords anglais et on ajoute des mots peu informatifs
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stop_words.update([
    "would", "could", "also", "may", "must", "one", "said",
    "like", "can", "us", "mr", "madam",
    "parliament", "commission", "european", "union"
])

# FONCTIONS TEXTE

# Fonction pour nettoyer le texte : minuscules, suppression des caractères non alphabétiques, espaces multiples
def nettoyer_texte(texte):
    texte = texte.lower()
    texte = re.sub(r"[^a-z\s]", " ", texte)
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()

# Fonction pour tokenizer le texte : on enlève les stopwords et mots trop courts
def tokenizer(texte):
    return [
        mot for mot in texte.split()
        if len(mot) > 3 and mot not in stop_words
    ]

# CHARGEMENT DU CORPUS

print(" Chargement du corpus EUPDCorp...")

documents = []  # On va stocker tous nos documents tokenisés ici

# On parcourt le corpus année par année
for dossier_annee in sorted(os.listdir(CHEMIN_CORPUS)):
    chemin_annee = os.path.join(CHEMIN_CORPUS, dossier_annee)
    if not os.path.isdir(chemin_annee):
        continue

    # On parcourt tous les fichiers texte de l'année
    for fichier in os.listdir(chemin_annee):
        if fichier.endswith(".txt"):
            chemin_fichier = os.path.join(chemin_annee, fichier)
            with open(chemin_fichier, encoding="utf-8", errors="ignore") as f:
                texte = f.read()

            # On nettoie le texte et on tokenize
            texte = nettoyer_texte(texte)
            tokens = tokenizer(texte)

            # On ne garde que les documents suffisamment longs
            if len(tokens) > 20:
                documents.append(tokens)

print(f" Documents analysés : {len(documents)}")

# LDA GLOBALE

print(" Lancement LDA globale...")

# On construit le dictionnaire et le corpus pour Gensim
dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(doc) for doc in documents]

# On entraîne le modèle LDA global
lda = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    passes=PASSES,
    random_state=RANDOM_STATE
)

# SAUVEGARDE DES RÉSULTATS

# On crée le dossier de sortie s’il n’existe pas
os.makedirs(DOSSIER_SORTIE, exist_ok=True)

txt_path = f"{DOSSIER_SORTIE}/lda_global_eupdcorp.txt"
csv_path = f"{DOSSIER_SORTIE}/lda_global_eupdcorp.csv"

rows = []  # On va stocker ici les résultats pour le CSV

# On écrit les topics dans un fichier TXT et on prépare le CSV
with open(txt_path, "w", encoding="utf-8") as f:
    for i, topic in lda.print_topics(num_words=TOP_MOTS):
        # On affiche chaque topic dans le terminale et dans le fichier
        ligne = f"Topic {i}: {topic}"
        print(ligne)
        f.write(ligne + "\n")

        # On stocke chaque mot et son poids pour le CSV
        for mot, poids in lda.show_topic(i, TOP_MOTS):
            rows.append({
                "topic": i,
                "mot": mot,
                "poids": poids
            })

# On sauvegarde les résultats dans un CSV pour analyse ultérieure
pd.DataFrame(rows).to_csv(csv_path, index=False)

print(" LDA globale terminée")
print(f"TXT : {txt_path}")
print(f" CSV : {csv_path}")

# SAUVEGARDE DU MODELE ET DICTIONNAIRE

# On crée le dossier models s’il n’existe pas
os.makedirs("models", exist_ok=True)

# On sauvegarde le dictionnaire et le modèle pour réutilisation future
dictionary.save("models/lda_ecologie.dict")
lda.save("models/lda_ecologie.model")