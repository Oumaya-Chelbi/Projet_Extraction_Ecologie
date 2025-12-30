# LDA PAR ANNÉE – ÉVOLUTION DU DISCOURS ÉCOLOGIQUE

import os
import re
from gensim import corpora
from gensim.models import LdaModel
import nltk
from nltk.corpus import stopwords
import pandas as pd

# PARAMÈTRES

# On définit le chemin vers notre corpus écologique filtré
CHEMIN_CORPUS = "data/processed/EUPDCorp_TXT_Ecologie1"

# Dossier où on va sauvegarder tous nos résultats
DOSSIER_SORTIE = "results/lda_ecologie_par_annee"

# Paramètres LDA : on choisit le nombre de topics, passes, mots clés etc.
NUM_TOPICS = 6
PASSES = 10
TOP_MOTS = 10
RANDOM_STATE = 42
MIN_TOKENS = 20  # On ne garde que les documents avec au moins ce nombre de tokens

# MOTS-CLÉS ÉCOLOGIQUES

# On établit une liste de mots clés liés à l'écologie pour suivre leur fréquence
MOTS_CLES_ECO = [
    "environment", "environmental", "climate", "global", "warming",
    "biodiversity", "ecosystem", "pollution", "sustainable",
    "sustainability", "renewable", "energy", "carbon",
    "greenhouse", "emissions", "ecological", "atmospheric",
    "ozone", "waste"
]

# STOPWORDS

# On télécharge les stopwords anglais et on complète avec des mots fréquents mais peu informatifs
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stop_words.update([
    "would", "could", "also", "may", "must", "one", "said",
    "like", "can", "us", "mr", "madam",
    "parliament", "commission", "european", "union"
])

# FONCTIONS TEXTE

# On définit une fonction pour nettoyer le texte
def nettoyer_texte(texte):
    # On passe tout en minuscules
    texte = texte.lower()
    # On retire tous les caractères non alphabétiques
    texte = re.sub(r"[^a-z\s]", " ", texte)
    # On réduit les espaces multiples
    texte = re.sub(r"\s+", " ", texte)
    # On retourne le texte nettoyé
    return texte.strip()

# Tokenizer simple : on découpe le texte en mots, on enlève les stopwords et mots trop courts
def tokenizer(texte):
    return [
        mot for mot in texte.split()
        if len(mot) > 3 and mot not in stop_words
    ]

# LDA PAR ANNÉE

# On crée le dossier de sortie s'il n'existe pas
os.makedirs(DOSSIER_SORTIE, exist_ok=True)

# On initialise un tableau pour stocker les résultats résumé
resume_rows = []

# On parcourt notre corpus année par année
for dossier_annee in sorted(os.listdir(CHEMIN_CORPUS)):
    chemin_annee = os.path.join(CHEMIN_CORPUS, dossier_annee)
    if not os.path.isdir(chemin_annee):
        continue

    print(f"LDA pour l'année {dossier_annee}")

    documents = []        # On stocke ici tous les documents valides pour l'année
    tokens_par_doc = []   # Et ici leurs tokens respectifs

    # On parcourt tous les fichiers texte de l'année
    for fichier in os.listdir(chemin_annee):
        if not fichier.endswith(".txt"):
            continue

        # On lit le fichier et on nettoie le texte
        with open(os.path.join(chemin_annee, fichier),
                  encoding="utf-8", errors="ignore") as f:
            texte = nettoyer_texte(f.read())
            tokens = tokenizer(texte)

            # On ne garde que les documents suffisamment longs
            if len(tokens) >= MIN_TOKENS:
                documents.append(tokens)
                tokens_par_doc.append(tokens)

    print(f"Documents retenus : {len(documents)}")

    # Si on a trop peu de documents, on ignore cette année
    if len(documents) < 5:
        print("Trop peu de documents, année ignorée")
        continue

    #  LDA 
    # On construit le dictionnaire et le corpus pour Gensim
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    # On entraîne le modèle LDA
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        passes=PASSES,
        random_state=RANDOM_STATE
    )

    #  Sortie par année 
    dossier_out_annee = os.path.join(DOSSIER_SORTIE, dossier_annee)
    os.makedirs(dossier_out_annee, exist_ok=True)

    rows_topics = []

    # On récupère les mots et poids de chaque topic
    for i, topic in lda.print_topics(num_words=TOP_MOTS):
        for mot, poids in lda.show_topic(i, TOP_MOTS):
            rows_topics.append({
                "annee": dossier_annee,
                "topic": i,
                "mot": mot,
                "poids": poids
            })

    # On sauvegarde les topics par année en CSV
    pd.DataFrame(rows_topics).to_csv(
        os.path.join(dossier_out_annee, f"lda_{dossier_annee}.csv"),
        index=False
    )

    #  POIDS DES TOPICS 
    # On calcule le poids total de chaque topic dans tous les documents
    topic_weights = {i: 0 for i in range(NUM_TOPICS)}

    for doc in corpus:
        for topic_id, weight in lda.get_document_topics(doc):
            topic_weights[topic_id] += weight

    # On compte combien de topics sont réellement actifs
    nb_topics_actifs = sum(1 for v in topic_weights.values() if v > 1)

    # On stocke les poids totaux pour le résumé
    for topic_id, poids_total in topic_weights.items():
        resume_rows.append({
            "annee": dossier_annee,
            "type": "topic_weight",
            "topic": topic_id,
            "valeur": poids_total
        })

    # Et le nombre de topics actifs
    resume_rows.append({
        "annee": dossier_annee,
        "type": "nb_topics_actifs",
        "topic": None,
        "valeur": nb_topics_actifs
    })

    #  FRÉQUENCE DES MOTS ÉCOLOGIQUES 
    # On compte combien de fois chaque mot-clé écologique apparaît
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

# On compile tous les résultats dans un seul CSV comparatif
df_resume = pd.DataFrame(resume_rows)
df_resume.to_csv(
    os.path.join(DOSSIER_SORTIE, "resume_comparatif_annees.csv"),
    index=False
)

print(" LDA par année + comparaison temporelle terminés")
