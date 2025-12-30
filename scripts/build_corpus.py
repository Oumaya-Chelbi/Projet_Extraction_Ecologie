# CONSTRUCTION DU CORPUS TXT - EUPDCorp (anglais)
# Agrégation par ANNÉE x PAYS

# On importe les bibliothèques nécessaires pour manipuler le CSV,
# gérer les dossiers et nettoyer le texte
import pandas as pd
import os
import re

# PARAMÈTRES

# Chemin vers le fichier CSV brut contenant l’ensemble des discours
CSV_PATH = "data/raw/EUPDCorp_1999-2024.csv"

# Dossier dans lequel on va enregistrer le corpus sous forme de fichiers TXT
DOSSIER_SORTIE = "data/EUPDCorp_TXT"

# Noms des colonnes utilisées dans le CSV
COL_DATE = "year"
COL_PAYS = "country"
COL_TEXTE = "speech_en"

# NETTOYAGE TEXTE

# Fonction de nettoyage du texte
# On met tout en minuscules et on supprime les espaces multiples
def nettoyer_texte(texte):
    texte = str(texte).lower()
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()

# CHARGEMENT DU CSV

# On charge le fichier CSV dans un DataFrame pandas
df = pd.read_csv(CSV_PATH)

# On affiche les colonnes détectées pour vérifier que les noms sont corrects
print("Colonnes détectées dans le CSV :")
print(df.columns.tolist())

# PRÉPARATION DES DONNÉES

# On crée des colonnes normalisées pour l’année, le pays et le texte
df["annee"] = df[COL_DATE].astype(str)
df["pays"] = df[COL_PAYS].astype(str)
df["texte"] = df[COL_TEXTE].astype(str).apply(nettoyer_texte)

# On supprime les lignes où le texte ou le pays est vide ou invalide
# afin d’éviter de créer des fichiers inutiles
df = df[
    (df["texte"] != "") &
    (df["texte"] != "nan") &
    (df["pays"] != "nan")
]

print(f"Nombre de tours de parole conservés : {len(df)}")

# CONSTRUCTION DU CORPUS TXT

# On crée le dossier de sortie s’il n’existe pas déjà
os.makedirs(DOSSIER_SORTIE, exist_ok=True)

# On regroupe les discours par année et par pays
groupes = df.groupby(["annee", "pays"])

# Pour chaque couple (année, pays), on crée un fichier texte
# contenant l’ensemble des discours correspondants
for (annee, pays), group in groupes:
    dossier_annee = os.path.join(DOSSIER_SORTIE, annee)
    os.makedirs(dossier_annee, exist_ok=True)

    chemin_fichier = os.path.join(dossier_annee, f"{pays}.txt")

    # On écrit tous les discours du pays pour l’année donnée
    # dans un seul fichier TXT
    with open(chemin_fichier, "w", encoding="utf-8") as f:
        for texte in group["texte"]:
            f.write(texte + "\n")

print("\nCorpus EUPDCorp construit avec succès")
print(f" Dossier de sortie : {DOSSIER_SORTIE}")
