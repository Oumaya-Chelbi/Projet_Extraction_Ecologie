# =====================================================
# FILTRAGE PAR PHRASES : CORPUS DE PASSAGES ÉCOLOGIQUES
# =====================================================

import os
import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

CORPUS_SOURCE = "data/processed/EUPDCorp_TXT"
CORPUS_ECO_PASSAGES = "data/processed/EUPDCorp_TXT_Ecologie1"

TERMES_ECOLOGIQUES = [
    "environment", "environmental", "climate", "global warming",
    "biodiversity", "ecosystem", "pollution", "sustainable",
    "sustainability", "renewable", "energy", "carbon",
    "greenhouse", "emissions", "ecological", "atmospheric", "ozone",
    "waste"
]

def nettoyer_texte(texte):
    return texte  # on garde le texte original ici pour ne pas casser la segmentation

def phrase_est_eco(phrase, lexique, min_termes=1):
    # version simple: lower + recherche de mots/expressions
    p = phrase.lower()
    # nettoyage léger pour éviter les faux positifs sur les mots collés
    p_norm = re.sub(r"[^a-z\s]", " ", p)
    p_norm = re.sub(r"\s+", " ", p_norm).strip()

    nb = 0
    for terme in lexique:
        # expressions multi-mots gérées par "in", mots simples avec \b
        if " " in terme:
            if terme in p_norm:
                nb += 1
        else:
            if re.search(rf"\b{re.escape(terme)}\b", p_norm):
                nb += 1
    return nb >= min_termes

print(" Construction du corpus écologiques...")

os.makedirs(CORPUS_ECO_PASSAGES, exist_ok=True)

nb_docs = 0
nb_docs_non_vides = 0

for dossier_annee in os.listdir(CORPUS_SOURCE):
    chemin_annee_src = os.path.join(CORPUS_SOURCE, dossier_annee)
    if not os.path.isdir(chemin_annee_src):
        continue

    chemin_annee_dst = os.path.join(CORPUS_ECO_PASSAGES, dossier_annee)
    os.makedirs(chemin_annee_dst, exist_ok=True)

    for fichier in os.listdir(chemin_annee_src):
        if not fichier.endswith(".txt"):
            continue

        nb_docs += 1
        chemin_fichier = os.path.join(chemin_annee_src, fichier)

        with open(chemin_fichier, encoding="utf-8", errors="ignore") as f:
            texte_brut = f.read()

        texte = nettoyer_texte(texte_brut)

        # segmentation en phrases
        phrases = sent_tokenize(texte)

        phrases_eco = [
            ph for ph in phrases
            if phrase_est_eco(ph, TERMES_ECOLOGIQUES, min_termes=1)
        ]

        if phrases_eco:
            nb_docs_non_vides += 1
            texte_filtre = "\n".join(phrases_eco)
            with open(os.path.join(chemin_annee_dst, fichier),
                      "w", encoding="utf-8") as out:
                out.write(texte_filtre)

print(f"Discours originaux : {nb_docs}")
print(f"Discours avec au moins un passage écologique : {nb_docs_non_vides}")
print(f"Dossier passages : {CORPUS_ECO_PASSAGES}")
