# Analyse descriptive + indicateurs écologiques
# Corpus EUPDCorp (TXT)

# On importe les bibliothèques nécessaires pour le traitement des données,
# le nettoyage des textes et la visualisation des résultats
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# PARAMÈTRES

# Chemin vers le corpus organisé par années et par pays
CORPUS_DIR = "data/processed/EUPDCorp_TXT"

# Dossier dans lequel on va sauvegarder les résultats de l’analyse descriptive
RESULTS_DIR = "results/descriptif"

# Lexique écologique simple qu'on a trouver grace au General
# Multilingual Environmental Thesaurus et Glossary of sustainability
# On utilise ici une liste de mots-clés fréquemment associés
# aux thématiques environnementales et climatiques
TERMES_ECOLOGIQUES = [
    "environment", "environmental", "climate", "climatic",
    "global warming", "pollution", "biodiversity",
    "ecosystem", "sustainable", "sustainability",
    "renewable", "energy", "greenhouse"
]

# On crée le dossier de résultats s’il n’existe pas déjà
os.makedirs(RESULTS_DIR, exist_ok=True)

# FONCTIONS

# Fonction de nettoyage du texte
# On met tout en minuscules et on supprime les caractères non alphabétiques
# afin de faciliter le comptage des mots
def nettoyer_texte(texte):
    texte = texte.lower()
    texte = re.sub(r"[^a-z\s]", " ", texte)
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()

# Fonction qui compte le nombre total de mots dans un texte
def compter_mots(texte):
    return len(texte.split())

# Fonction qui compte le nombre d’occurrences des termes écologiques
# définis dans le lexique
def compter_mots_ecologiques(texte):
    return sum(texte.count(mot) for mot in TERMES_ECOLOGIQUES)

# CHARGEMENT DU CORPUS

print(" Lecture du corpus...")

# Liste qui va contenir toutes les informations extraites des discours
donnees = []

# On parcourt les dossiers correspondant aux années
for annee in os.listdir(CORPUS_DIR):
    chemin_annee = os.path.join(CORPUS_DIR, annee)

    # On vérifie qu’il s’agit bien d’un dossier
    if not os.path.isdir(chemin_annee):
        continue

    # On parcourt les fichiers texte de chaque année
    for fichier in os.listdir(chemin_annee):
        if not fichier.endswith(".txt"):
            continue

        # On extrait le pays à partir du nom du fichier
        pays = fichier.split("_")[0]
        chemin_fichier = os.path.join(chemin_annee, fichier)

        # On lit et nettoie le texte du discours
        with open(chemin_fichier, encoding="utf-8", errors="ignore") as f:
            texte = nettoyer_texte(f.read())

        # On calcule le nombre total de mots
        nb_mots = compter_mots(texte)

        # On calcule le nombre de mots écologiques
        nb_mots_eco = compter_mots_ecologiques(texte)

        # On stocke les informations
        donnees.append({
            "annee": int(float(annee)),
            "pays": pays,
            "nb_mots": nb_mots,
            "nb_mots_ecologiques": nb_mots_eco
        })

# On convertit les données en DataFrame pour faciliter l’analyse
df = pd.DataFrame(donnees)

print(f" {len(df)} discours analysés")

# STATISTIQUES GLOBALES

# Calcul des indicateurs généraux du corpus
nb_textes_total = len(df)
nb_mots_total = df["nb_mots"].sum()
nb_mots_moyen = df["nb_mots"].mean()
nb_pays = df["pays"].nunique()
nb_annees = df["annee"].nunique()

# Nombre de discours par année
discours_par_annee = df.groupby("annee").size()

# Nombre de discours par pays
discours_par_pays = df.groupby("pays").size()

# ANALYSE ÉCOLOGIQUE

# On sélectionne uniquement les discours contenant au moins un mot écologique
df_eco = df[df["nb_mots_ecologiques"] > 0]

# On identifie la première année d’apparition des termes écologiques
premiere_annee_eco = df_eco["annee"].min()

# On calcule l’évolution annuelle des mots totaux et des mots écologiques
evolution_eco = df.groupby("annee").agg(
    mots_totaux=("nb_mots", "sum"),
    mots_ecologiques=("nb_mots_ecologiques", "sum")
)

# On calcule la proportion de mots écologiques par rapport au volume total
evolution_eco["proportion_ecologique"] = (
    evolution_eco["mots_ecologiques"] / evolution_eco["mots_totaux"]
)

# AFFICHAGE DANS LE TERMINALE

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

# On sauvegarde les résultats sous forme de fichiers CSV
discours_par_annee.to_csv(f"{RESULTS_DIR}/discours_par_annee.csv")
discours_par_pays.to_csv(f"{RESULTS_DIR}/discours_par_pays.csv")
evolution_eco.to_csv(f"{RESULTS_DIR}/evolution_ecologie.csv")
df.to_csv(f"{RESULTS_DIR}/df_complet.csv", index=False)

# GRAPHIQUE : ÉVOLUTION DES MOTS ÉCOLOGIQUES

# On crée un graphique montrant l’évolution du volume de mots écologiques
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
