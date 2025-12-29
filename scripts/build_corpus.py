# =====================================================
# CONSTRUCTION DU CORPUS TXT - EUPDCorp (anglais)
# Agrégation par ANNÉE x PAYS
# =====================================================

import pandas as pd
import os
import re

# -----------------------------------------------------
# PARAMÈTRES
# -----------------------------------------------------

CSV_PATH = "data/raw/EUPDCorp_1999-2024.csv"
DOSSIER_SORTIE = "data/EUPDCorp_TXT"

COL_DATE = "year"
COL_PAYS = "country"
COL_TEXTE = "speech_en"

# -----------------------------------------------------
# NETTOYAGE TEXTE
# -----------------------------------------------------

def nettoyer_texte(texte):
    texte = str(texte).lower()
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()

# -----------------------------------------------------
# CHARGEMENT DU CSV
# -----------------------------------------------------

df = pd.read_csv(CSV_PATH)

print("Colonnes détectées dans le CSV :")
print(df.columns.tolist())

# -----------------------------------------------------
# PRÉPARATION DES DONNÉES
# -----------------------------------------------------

df["annee"] = df[COL_DATE].astype(str)
df["pays"] = df[COL_PAYS].astype(str)
df["texte"] = df[COL_TEXTE].astype(str).apply(nettoyer_texte)

# Suppression lignes vides
df = df[
    (df["texte"] != "") &
    (df["texte"] != "nan") &
    (df["pays"] != "nan")
]

print(f"Nombre de tours de parole conservés : {len(df)}")

# -----------------------------------------------------
# CONSTRUCTION DU CORPUS TXT
# -----------------------------------------------------

os.makedirs(DOSSIER_SORTIE, exist_ok=True)

groupes = df.groupby(["annee", "pays"])

for (annee, pays), group in groupes:
    dossier_annee = os.path.join(DOSSIER_SORTIE, annee)
    os.makedirs(dossier_annee, exist_ok=True)

    chemin_fichier = os.path.join(dossier_annee, f"{pays}.txt")

    with open(chemin_fichier, "w", encoding="utf-8") as f:
        for texte in group["texte"]:
            f.write(texte + "\n")

print("\nCorpus EUPDCorp construit avec succès")
print(f" Dossier de sortie : {DOSSIER_SORTIE}")
