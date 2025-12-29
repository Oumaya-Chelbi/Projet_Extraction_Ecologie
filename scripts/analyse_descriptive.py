# =====================================================
# analyse_descriptive.py
# Analyse descriptive + indicateurs écologiques
# Corpus EUPDCorp (TXT)
# =====================================================

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# PARAMÈTRES

CORPUS_DIR = "data/processed/EUPDCorp_TXT"
RESULTS_DIR = "results/descriptif"

# lexique écologique SIMPLE et justifiable
TERMES_ECOLOGIQUES = [
    "environment", "environmental", "climate", "climatic",
    "global warming", "pollution", "biodiversity",
    "ecosystem", "sustainable", "sustainability",
    "renewable", "energy", "greenhouse"
]

os.makedirs(RESULTS_DIR, exist_ok=True)

# FONCTIONS

def nettoyer_texte(texte):
    texte = texte.lower()
    texte = re.sub(r"[^a-z\s]", " ", texte)
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()

def compter_mots(texte):
    return len(texte.split())

def compter_mots_ecologiques(texte):
    return sum(texte.count(mot) for mot in TERMES_ECOLOGIQUES)

# CHARGEMENT DU CORPUS

print(" Lecture du corpus...")

donnees = []

for annee in os.listdir(CORPUS_DIR):
    chemin_annee = os.path.join(CORPUS_DIR, annee)

    if not os.path.isdir(chemin_annee):
        continue

    for fichier in os.listdir(chemin_annee):
        if not fichier.endswith(".txt"):
            continue

        pays = fichier.split("_")[0]
        chemin_fichier = os.path.join(chemin_annee, fichier)

        with open(chemin_fichier, encoding="utf-8", errors="ignore") as f:
            texte = nettoyer_texte(f.read())

        nb_mots = compter_mots(texte)
        nb_mots_eco = compter_mots_ecologiques(texte)

        donnees.append({
            "annee": int(float(annee)),
            "pays": pays,
            "nb_mots": nb_mots,
            "nb_mots_ecologiques": nb_mots_eco
        })

df = pd.DataFrame(donnees)

print(f" {len(df)} discours analysés")

# STATISTIQUES GLOBALES

nb_textes_total = len(df)
nb_mots_total = df["nb_mots"].sum()
nb_mots_moyen = df["nb_mots"].mean()
nb_pays = df["pays"].nunique()
nb_annees = df["annee"].nunique()

discours_par_annee = df.groupby("annee").size()
discours_par_pays = df.groupby("pays").size()

# ANALYSE ÉCOLOGIQUE

df_eco = df[df["nb_mots_ecologiques"] > 0]

premiere_annee_eco = df_eco["annee"].min()

evolution_eco = df.groupby("annee").agg(
    mots_totaux=("nb_mots", "sum"),
    mots_ecologiques=("nb_mots_ecologiques", "sum")
)
evolution_eco["proportion_ecologique"] = (
    evolution_eco["mots_ecologiques"] / evolution_eco["mots_totaux"]
)

# AFFICHAGE CONSOLE

print("\n===== STATISTIQUES GLOBALES =====")
print(f"Nombre total de textes : {nb_textes_total}")
print(f"Nombre total de mots : {nb_mots_total}")
print(f"Nombre moyen de mots par discours : {nb_mots_moyen:.2f}")
print(f"Nombre de pays distincts : {nb_pays}")
print(f"Nombre d’années couvertes : {nb_annees}")

print("\n===== DISCOURS PAR ANNÉE =====")
print(discours_par_annee.head())

print("\n===== DISCOURS PAR PAYS (Top 10) =====")
print(discours_par_pays.sort_values(ascending=False).head(10))

print("\n===== ANALYSE ÉCOLOGIQUE =====")
print(f"Première année d’apparition des termes écologiques : {premiere_annee_eco}")

# SAUVEGARDE DES RÉSULTATS

discours_par_annee.to_csv(f"{RESULTS_DIR}/discours_par_annee.csv")
discours_par_pays.to_csv(f"{RESULTS_DIR}/discours_par_pays.csv")
evolution_eco.to_csv(f"{RESULTS_DIR}/evolution_ecologie.csv")
df.to_csv(f"{RESULTS_DIR}/df_complet.csv", index=False)

# GRAPHIQUE : ÉVOLUTION DES MOTS ÉCOLOGIQUES

plt.figure(figsize=(10, 5))
plt.plot(evolution_eco.index, evolution_eco["mots_ecologiques"], marker="o")
plt.xlabel("Année")
plt.ylabel("Nombre de mots écologiques")
plt.title("Évolution du volume de mots écologiques dans le temps (EUPDCorp)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/evolution_mots_ecologiques.png")
plt.show()

print("Analyse descriptive terminée")
print(f"Résultats dans : {RESULTS_DIR}")
