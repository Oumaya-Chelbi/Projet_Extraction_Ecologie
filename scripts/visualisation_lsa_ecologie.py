import os
import pandas as pd
import matplotlib.pyplot as plt

# PARAMÈTRES

DOSSIER_RESULTATS = "results/lsa_pays_ecologie"
DOSSIER_FIGURES = "results/figures_lsa"
os.makedirs(DOSSIER_FIGURES, exist_ok=True)

# CHARGEMENT DES DONNÉES

df_docs = pd.read_csv(
    os.path.join(DOSSIER_RESULTATS, "scores_ecologiques_lsa_par_discours.csv")
)

df_top10 = pd.read_csv(
    os.path.join(DOSSIER_RESULTATS, "top10_pays_et_annee_max_lsa_ecologie.csv")
)

# GRAPHIQUE 2
# ÉVOLUTION TEMPORELLE — TOP 5 PAYS

# Identifier les 5 pays les plus écologiques (score total)
top5_pays = (
    df_top10
    .sort_values("score_eco_total_pays", ascending=False)
    .head(5)["pays"]
    .tolist()
)

# Agrégation par pays et par année
df_evol = (
    df_docs[df_docs["pays"].isin(top5_pays)]
    .groupby(["pays", "annee"], as_index=False)
    ["score_eco"]
    .sum()
)

plt.figure(figsize=(9, 5))

for pays in top5_pays:
    df_p = df_evol[df_evol["pays"] == pays]
    plt.plot(
        df_p["annee"],
        df_p["score_eco"],
        marker="o",
        label=pays
    )

plt.xlabel("Année")
plt.ylabel("Score écologique annuel (LSA)")
plt.title("Évolution du discours écologique (LSA)\nTop 5 pays les plus écologiques")
plt.legend(title="Pays")
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(DOSSIER_FIGURES, "evolution_top5_pays_lsa.png"))
plt.close()

# GRAPHIQUE 3
# ANNÉE DE PIC ÉCOLOGIQUE — TOP 10

plt.figure(figsize=(8, 5))

plt.scatter(
    df_top10["annee_max_eco"],
    df_top10["score_eco_max_annee"],
)

# Annotation des points (pays)
for _, row in df_top10.iterrows():
    plt.annotate(
        row["pays"],
        (row["annee_max_eco"], row["score_eco_max_annee"]),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=9
    )

plt.xlabel("Année du pic écologique")
plt.ylabel("Score écologique maximal (LSA)")
plt.title("Année de pic du discours écologique par pays (LSA)")
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(DOSSIER_FIGURES, "pic_ecologique_top10_lsa.png"))
plt.close()

print(" Graphiques LSA générés avec succès :")
print(" evolution_top5_pays_lsa.png")
print(" pic_ecologique_top10_lsa.png")
