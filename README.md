# Projet_Extraction_Ecologie

Ce dépôt contient un projet d’analyse automatique des discours du Parlement européen (corpus EUPDCorp, 1999–2024) pour détecter, mesurer et suivre les thématiques écologiques (climat, environnement, énergie, durabilité, etc.) selon les années et les pays.


## Objectifs

Construire un corpus textuel propre, organisé par année et par pays, à partir d’un fichier brut de discours parlementaires.

Extraire un sous‑corpus écologique en ne conservant que les passages qui parlent d’environnement et de climat, à l’aide d’un lexique dédié.

Appliquer des modèles thématiques (LDA) et des modèles à espaces latents (LSA) sur ce sous‑corpus écologique et comparer les deux.

Quantifier l’intensité du discours écologique dans le temps (par année) et selon les pays (score global, top‑10 pays, année la plus « écologique » par pays).

Produire des visualisations montrant l’émergence et l’évolution des thèmes écologiques dans les débats parlementaires.

## Organisation du dépôt

**data/**
raw/ : données brutes non versionnées (ex. EUPDCorp_1999-2024.csv, trop volumineux pour GitHub).
TXT_UNGDC : initialement on allait travailler avec ce corpus mais on a changé car celui-ci ne mentionne que très très peu l'écologie.
processed/ :
EUPDCorp_TXT/ANNEE/PAYS.txt : corpus texte nettoyé, un fichier par combinaison année × pays.
EUPDCorp_TXT_Ecologie1/ : sous‑corpus écologique construit par filtrage de phrases contenant des termes écologiques.

**scripts/**

build_corpus.py : construit EUPDCorp_TXT à partir du CSV brut (agrégation par année/pays, nettoyage léger).

build_corpus_ecologie.py : crée EUPDCorp_TXT_Ecologie1 en ne gardant que les phrases contenant au moins un terme du lexique écologique.

analyse_descriptive.py : statistiques descriptives sur le corpus global (discours par année/pays, volume de mots écologiques, premières années d’apparition, etc.).

lda_globale.py :
Première execution sur le corpus initiale EUPDCorp_TXT (sortie dans results/lda/).

Deuxième execution : LDA sur le sous‑corpus écologique (EUPDCorp_TXT_Ecologie1), production de 10 topics écologiques + sauvegarde du modèle et du dictionnaire (sortie dans results/lda_ecologie/ et models/).

lda_annee.py : LDA par année sur le sous‑corpus écologique, plus calcul de poids de topics et de fréquences de mots clés par année.

lda_pays.py : application du modèle LDA écologique au corpus global pour obtenir un score écologique par discours, puis agrégation par pays et par année (classement des pays + année la plus écologique par pays).

lsa_eco.py : apprentissage d’un modèle TF‑IDF + TruncatedSVD (LSA) sur le sous‑corpus écologique.

lsa_pays_annee.py : projection des discours du corpus global dans l’espace LSA écologique, score écologique LSA par discours, agrégation par pays et par année.

visualisation_lda_ecologie.py : visualisations de l’intensité, de la diversité et de l’évolution temporelle des thèmes écologiques (à partir des sorties LDA par année).

**results/**

descriptif/ : CSV de synthèse (discours par année, par pays, évolution des mots écologiques) et graphiques descriptifs.

lda/ : résultats LDA globaux (topics sur tout le corpus ).

lda_ecologie/ : topics LDA sur le sous‑corpus écologique (fichiers texte/CSV).

lda_ecologie_par_annee/ : résultats LDA par année + fichiers de résumé comparatif.

lda_pays_ecologie/ : scores écologiques LDA par discours, classement des pays, top‑10 pays + année max.

lsa_pays_ecologie/ : scores écologiques LSA par discours, classement LSA des pays, top‑10 pays + année max.

visualisations_ecologie/ : figures (intensité écologique, diversité des topics, évolution des mots‑clés, évolution des mots dominants LDA).


**models/**

lda_ecologie.model, lda_ecologie.dict, et fichiers associés : modèle LDA écologique Gensim et dictionnaire utilisés pour scorer les discours et construire les classements par pays/année.

**models_lsa/**

tfidf_eco.pkl, lsa_eco.pkl, lsa_eco_top_terms.txt : vectoriseur TF‑IDF et modèle LSA entraînés sur le sous‑corpus écologique, plus liste des termes dominants par composante.

**requirements.txt**

Liste des dépendances Python nécessaires (par exemple : pandas, numpy, gensim, scikit-learn, nltk, matplotlib, etc.).
