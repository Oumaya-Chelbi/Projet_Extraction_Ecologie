# =====================================================
# SCRIPT 2 : LSA ÉCOLOGIQUE SUR LES PASSAGES ÉCO
# =====================================================

import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib
import numpy as np

# Paramètres

CORPUS_ECO_PASSAGES = "data/processed/EUPDCorp_TXT_Ecologie1"
MODELS_DIR = "models_lsa"
N_COMPONENTS = 20   # dim LSA 

os.makedirs(MODELS_DIR, exist_ok=True)

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

def charger_textes_corpus(dossier_racine):
    textes = []
    for dossier_annee in sorted(os.listdir(dossier_racine)):
        chemin_annee = os.path.join(dossier_racine, dossier_annee)
        if not os.path.isdir(chemin_annee):
            continue
        for fichier in os.listdir(chemin_annee):
            if not fichier.endswith(".txt"):
                continue
            chemin_fichier = os.path.join(chemin_annee, fichier)
            with open(chemin_fichier, encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            txt = nettoyer_texte(txt)
            if txt.strip():
                textes.append(txt)
    return textes

print(" Chargement des passages écologiques...")
textes_eco = charger_textes_corpus(CORPUS_ECO_PASSAGES)
print(f"Nombre de documents (passages éco) : {len(textes_eco)}")

if not textes_eco:
    raise ValueError(f"Aucun document trouvé dans {CORPUS_ECO_PASSAGES}")

print("Construction TF-IDF...")
vectorizer = TfidfVectorizer(
    stop_words=list(stop_words),
    min_df=2   # mets 1 si corpus très petit
)
X_eco = vectorizer.fit_transform(textes_eco)

if X_eco.shape[1] < N_COMPONENTS:
    # sécurité si peu de termes par rapport à N_COMPONENTS
    N_COMPONENTS = max(2, X_eco.shape[1] // 2)
    print(f"⚠️ Peu de termes ({X_eco.shape[1]}), N_COMPONENTS réduit à {N_COMPONENTS}")

print(" Apprentissage LSA (TruncatedSVD)...")
svd = TruncatedSVD(
    n_components=N_COMPONENTS,
    random_state=42
)
X_eco_lsa = svd.fit_transform(X_eco)

# Sauvegarde des modèles
joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_eco.pkl"))
joblib.dump(svd, os.path.join(MODELS_DIR, "lsa_eco.pkl"))

print(" Modèles TF-IDF et LSA sauvegardés dans", MODELS_DIR)

# Affichage rapide des top termes par composante pour interprétation
terms = np.array(vectorizer.get_feature_names_out())

top_terms_path = os.path.join(MODELS_DIR, "lsa_eco_top_terms.txt")
with open(top_terms_path, "w", encoding="utf-8") as f:
    for i, comp in enumerate(svd.components_):
        idx = np.argsort(comp)[::-1][:15]
        top_terms = terms[idx]
        ligne = f"Component {i}: " + ", ".join(top_terms)
        print(ligne)
        f.write(ligne + "\n")

print(f" Top termes par composante enregistrés dans : {top_terms_path}")
