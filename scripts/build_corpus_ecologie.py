# FILTRAGE PAR PHRASES : CORPUS DE PASSAGES ÉCOLOGIQUES

# On importe les bibliothèques nécessaires pour parcourir les dossiers,
# nettoyer le texte et segmenter les discours en phrases
import os
import re
import nltk
from nltk.tokenize import sent_tokenize

# Téléchargement du tokenizer de phrases de NLTK
# Il est nécessaire pour découper correctement les discours en phrases
nltk.download("punkt")

# Dossier source : corpus complet organisé par année et par pays
CORPUS_SOURCE = "data/processed/EUPDCorp_TXT"

# Dossier de sortie : corpus ne contenant que les passages écologiques
CORPUS_ECO_PASSAGES = "data/processed/EUPDCorp_TXT_Ecologie1"

# Lexique écologique utilisé pour repérer les phrases liées à l’écologie
# Il s’agit d’un lexique simple, composé de termes fréquents dans les discours environnementaux
# qu'on a trouver grace au General Multilingual Environmental Thesaurus et Glossary of sustainability
TERMES_ECOLOGIQUES = [
    "environment", "environmental", "climate", "global warming",
    "biodiversity", "ecosystem", "pollution", "sustainable",
    "sustainability", "renewable", "energy", "carbon",
    "greenhouse", "emissions", "ecological", "atmospheric", "ozone",
    "waste"
]

# Fonction qui détermine si une phrase est écologique ou non
# Une phrase est considérée comme écologique si elle contient au moins
# un terme du lexique écologique
def phrase_est_eco(phrase, lexique, min_termes=1):
    # On met la phrase en minuscules pour éviter les problèmes de casse
    p = phrase.lower()

    # Nettoyage léger pour éviter les faux positifs liés à la ponctuation
    p_norm = re.sub(r"[^a-z\s]", " ", p)
    p_norm = re.sub(r"\s+", " ", p_norm).strip()

    nb = 0
    for terme in lexique:
        # On gère les expressions composées (ex: "global warming")
        # et les mots simples avec des bornes lexicales
        if " " in terme:
            if terme in p_norm:
                nb += 1
        else:
            if re.search(rf"\b{re.escape(terme)}\b", p_norm):
                nb += 1

    # La phrase est considérée comme écologique si le nombre de termes
    # détectés dépasse le seuil fixé
    return nb >= min_termes

print(" Construction du corpus écologiques...")

# Création du dossier de sortie s’il n’existe pas déjà
os.makedirs(CORPUS_ECO_PASSAGES, exist_ok=True)

# Compteurs pour suivre le nombre de fichiers traités
nb_docs = 0
nb_docs_non_vides = 0

# On parcourt le corpus source année par année
for dossier_annee in os.listdir(CORPUS_SOURCE):
    chemin_annee_src = os.path.join(CORPUS_SOURCE, dossier_annee)
    if not os.path.isdir(chemin_annee_src):
        continue

    # On crée le dossier correspondant dans le corpus écologique
    chemin_annee_dst = os.path.join(CORPUS_ECO_PASSAGES, dossier_annee)
    os.makedirs(chemin_annee_dst, exist_ok=True)

    # On parcourt les fichiers texte de chaque pays
    for fichier in os.listdir(chemin_annee_src):
        if not fichier.endswith(".txt"):
            continue

        nb_docs += 1
        chemin_fichier = os.path.join(chemin_annee_src, fichier)

        # Lecture du discours complet
        with open(chemin_fichier, encoding="utf-8", errors="ignore") as f:
            texte_brut = f.read()

        # Segmentation du discours en phrases
        phrases = sent_tokenize(texte_brut)

        # Sélection uniquement des phrases contenant des termes écologiques
        phrases_eco = [
            ph for ph in phrases
            if phrase_est_eco(ph, TERMES_ECOLOGIQUES, min_termes=1)
        ]

        # Si au moins une phrase écologique est trouvée,
        # on crée un fichier contenant uniquement ces passages
        if phrases_eco:
            nb_docs_non_vides += 1
            texte_filtre = "\n".join(phrases_eco)
            with open(os.path.join(chemin_annee_dst, fichier),
                      "w", encoding="utf-8") as out:
                out.write(texte_filtre)

# Affichage des statistiques finales
print(f"Discours originaux : {nb_docs}")
print(f"Discours avec au moins un passage écologique : {nb_docs_non_vides}")
print(f"Dossier passages : {CORPUS_ECO_PASSAGES}")
