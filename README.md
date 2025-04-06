# Projet Node2Vec — Prédiction et Clustering sur Graphes Réels et Simulés

## Objectif

Dans ce projet, j’ai souhaité **tester l’algorithme Node2Vec** sur **deux types de graphes** :

- **Un graphe simulé**, représentant une **propagation d’un virus (type COVID-19)** dans une population.  
  Ce cas permet de **prédire les individus infectés** à partir de leurs interactions spatio-temporelles.
- **Un graphe réel**, le célèbre **réseau de coapparitions des personnages du roman _Les Misérables_**.  
  Ce cas permet d’**analyser la structure du graphe** et de tester la **qualité des embeddings via clustering**.

L’objectif global est de **générer des représentations vectorielles (embeddings)** à l’aide de **Node2Vec** (modèle Skip-Gram avec negative sampling), et de les exploiter pour :

- Réaliser une **classification supervisée** (ex : prédire les infectés COVID).
- Appliquer un **clustering non supervisé** (ex : détecter des groupes dans Les Misérables).
- Évaluer la qualité des embeddings via des **métriques comme ARI et NMI**.


---

## 1. Dataset simulé : Propagation d'une infection

### Génération des données : `generate_data_propagation.py`

- Simulation de **80 individus** se déplaçant dans un espace géographique et temporel.
- Calcul des distances spatio-temporelles entre individus (via la **formule de Haversine**).
- Construction d’un **graphe** où chaque nœud est un individu.
- Propagation aléatoire de l'infection à partir de **3 sources**.
- Sauvegarde des données : `node2vecgps_data_covid_sparse.csv`.

### Chaque nœud est labellisé :

- 🟢 `infected = False` (non infecté)  
- 🔴 `infected = True` (infecté)

---

## 2. Construction du graphe : `build_graph_sparse.py`

- Utilisation des données simulées pour construire un graphe **NetworkX**.
- Création des arêtes si deux personnes sont proches dans **l’espace et le temps**.
- Sauvegarde du graphe : `graph.pkl`.

---

##  3. Entraînement Node2Vec : `train_node2vec.py`

### Fonctionnement :
- Génération de **random walks** sur le graphe.
- Création de **paires SkipGram** (nœud central + contexte).
- Apprentissage via un modèle **SkipGram + negative sampling** (voir `utils.py`).
- Optimisation automatique des **hyperparamètres** avec **Optuna**.

### Sauvegardes :
- `embeddings_optimized.pt` : embeddings appris
- `loss_values.txt` : valeurs de perte par époque

---

## 4. Visualisation & Évaluation : `viz_result.py`

- Visualisation des embeddings avec **t-SNE** et **PCA**.
- Couleur des points :
  - 🔴 rouge = infecté
  - 🟢 vert = non infecté
- **Classification binaire** (infecté vs non infecté) avec une régression logistique.
- Courbe d'apprentissage enregistrée dans : `loss_curve_reloaded.png`

---

##  5. Tests unitaires : `test_node2vec.py`

Tests automatisés avec **pytest** pour valider :

- Génération des fichiers (`embeddings`, `loss`)
- Format et validité des embeddings (pas de NaN/Inf)
- Diminution progressive de la perte
- Performance minimale de classification (> 50%)

---

## 6. Application sur le graphe réel **Les Misérables**

- Chargement d’un graphe réel avec **77 personnages**.
- Entraînement **Node2Vec** sur ce graphe **non-labellisé**.
- Ajout automatique d’un champ **`group`** via **détection de communautés (Louvain)**.
- Clustering `KMeans` appliqué aux embeddings.


---

## Résultats produits

| Fichier | Description |
|--------|-------------|
| `embeddings_optimized.pt` | Embeddings Node2Vec du graphe |
| `loss_values.txt`         | Historique des pertes par époque |
| `tsne_miserables.png`     | Visualisation t-SNE |
| `pca_miserables.png`      | Visualisation PCA |
| `clustering_scores.txt`   | Scores ARI et NMI |
| `classification_accuracy.txt` | Performance du classifieur (simulé) |

---

## Conclusion

Ce projet montre l'efficacité de **Node2Vec** pour représenter des graphes sous forme vectorielle.  
Il a été testé sur :

- Un graphe **simulé** avec propagation d'infection
- Un graphe **réel** : *Les Misérables*

Les performances ont été évaluées via :

- Des visualisations (t-SNE, PCA)
- Des métriques de classification
- Des scores de clustering (ARI, NMI)

---
Ce projet a été réalisé par : Donia Tekaya/ Mohamed Elyes Maalel / Mohamed Aziz Marouani
