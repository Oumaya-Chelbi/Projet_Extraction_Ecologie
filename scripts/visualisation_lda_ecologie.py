# VISUALISATIONS LDA – DISCOURS ÉCOLOGIQUE (UE)

import os
import pandas as pd
import matplotlib.pyplot as plt

# CHEMINS DES FICHIERS

DOSSIER_LDA = "results/lda_ecologie_par_annee"
FICHIER_TOPICS = os.path.join(DOSSIER_LDA, "lda_topics_par_annee.csv")
FICHIER_RESUME = os.path.join(DOSSIER_LDA, "resume_comparatif_annees.csv")

# Dossier où on va sauvegarder nos visualisations
DOSSIER_SORTIE = "results/visualisations_ecologie"
os.makedirs(DOSSIER_SORTIE, exist_ok=True)

# CHARGEMENT DES DONNÉES

print(" Chargement des données...")

# On lit les fichiers CSV générés par scripts LDA
df_topics = pd.read_csv(FICHIER_TOPICS)
df_resume = pd.read_csv(FICHIER_RESUME)

# On s'assure que les années sont bien en float
df_topics["annee"] = df_topics["annee"].astype(float)
df_resume["annee"] = df_resume["annee"].astype(float)

print(" Données chargées")

# INTENSITÉ ÉCOLOGIQUE (POIDS DES TOPICS)

# On calcule le poids total des topics écologiques par année
df_intensite = (
    df_resume[df_resume["type"] == "topic_weight"]
    .groupby("annee")["valeur"]
    .sum()
    .reset_index(name="poids_ecologique_total")
)

plt.figure(figsize=(10, 5))
plt.plot(
    df_intensite["annee"],
    df_intensite["poids_ecologique_total"],
    marker="o"
)

plt.xlabel("Année")
plt.ylabel("Poids total des topics écologiques")
plt.title("Évolution de l’intensité du discours écologique")
plt.grid(True)
plt.tight_layout()

# On sauvegarde la figure et on l'affiche
path1 = os.path.join(DOSSIER_SORTIE, "intensite_ecologique.png")
plt.savefig(path1)
plt.show()

# DIVERSITÉ THÉMATIQUE

# On regarde le nombre de topics actifs par année
# NB : topic actif = topic qui apparaît avec un poids non nul dans au moins un document de cette année.
df_diversite = (
    df_resume[df_resume["type"] == "nb_topics_actifs"]
    .sort_values("annee")
)

plt.figure(figsize=(10, 5))
plt.bar(
    df_diversite["annee"],
    df_diversite["valeur"]
)

plt.xlabel("Année")
plt.ylabel("Nombre de topics actifs")
plt.title("Diversité des thèmes écologiques par année")
plt.grid(axis="y")
plt.tight_layout()

path2 = os.path.join(DOSSIER_SORTIE, "diversite_topics.png")
plt.savefig(path2)
plt.show()

# ÉVOLUTION DES MOTS-CLÉS ÉCOLOGIQUES

df_mots = df_resume[df_resume["type"] == "mot_cle_eco"]

# On fait un pivot pour visualiser la fréquence des mots par année
pivot_mots = df_mots.pivot_table(
    index="annee",
    columns="topic",
    values="valeur",
    aggfunc="sum",
    fill_value=0
)

# On trace l'évolution de tous les mots-clés écologiques
pivot_mots.plot(figsize=(12, 6), marker="o")

plt.xlabel("Année")
plt.ylabel("Fréquence")
plt.title("Évolution des mots-clés écologiques")
plt.grid(True)
plt.tight_layout()

path3 = os.path.join(DOSSIER_SORTIE, "evolution_mots_cles.png")
plt.savefig(path3)
plt.show()

# MOTS DOMINANTS LDA

# On identifie les 6 mots les plus dominants dans le modèle LDA
top_mots = (
    df_topics
    .groupby("mot")["poids"]
    .sum()
    .sort_values(ascending=False)
    .head(6)
    .index
)

df_top = df_topics[df_topics["mot"].isin(top_mots)]

# Pivot pour visualiser l'évolution des poids de ces mots
pivot_lda = df_top.pivot_table(
    index="annee",
    columns="mot",
    values="poids",
    aggfunc="sum",
    fill_value=0
)

# On trace l'évolution de ces mots dominants
pivot_lda.plot(figsize=(12, 6), marker="o")

plt.xlabel("Année")
plt.ylabel("Poids LDA")
plt.title("Évolution des mots écologiques dominants (LDA)")
plt.grid(True)
plt.tight_layout()

path4 = os.path.join(DOSSIER_SORTIE, "evolution_mots_lda.png")
plt.savefig(path4)
plt.show()

print("Visualisations terminées")
