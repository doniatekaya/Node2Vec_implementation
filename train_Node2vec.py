# ============================
# Node2Vec - Entraînement optimisé avec Optuna 
# ============================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import networkx as nx
import random
import pickle
import optuna
import matplotlib
matplotlib.use("Agg")  # Forcer la génération de figures dans un environnement sans GUI
import os
import pickle as pkl
from utils import SkipGramModel, generate_walks, generate_skipgram_pairs, negative_sampling

# 1. Chargement du graphe de contact
with open("graph.pkl", "rb") as f:
    G = pickle.load(f)

print("Graphe chargé :", G.number_of_nodes(), "nœuds -", G.number_of_edges(), "arretes")

# 2. Encodage string to int pour les noeuds
"""Encode les nœuds avec des entiers (0, 1, 2,...) pour être compatibles avec PyTorch. """
node2idx = {node: idx for idx, node in enumerate(G.nodes())}
idx2node = {idx: node for node, idx in node2idx.items()}
G = nx.relabel_nodes(G, node2idx)

# 3. Objectif d'optimisation (Optuna)
def objective(trial):
    embedding_dim = trial.suggest_categorical("embedding_dim", [16, 32, 64])
    walk_length = trial.suggest_categorical("walk_length", [10, 20])
    window_size = trial.suggest_categorical("window_size", [3, 5])
    num_negatives = trial.suggest_categorical("num_negatives", [5, 10])
    learning_rate = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    num_walks = 10
    epochs = 10

    walks = generate_walks(G, num_walks=num_walks, walk_length=walk_length)
    pairs = generate_skipgram_pairs(walks, window_size=window_size)

    model = SkipGramModel(len(G.nodes), embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sampler = negative_sampling(G, len(G.nodes), num_negatives)

    model.train()
    """ À chaque époque, le modèle est entraîné sur des batches (positifs + négatifs). 
    On retourne la loss moyenne de la dernière époque pour qu’Optuna juge la qualité."""
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(pairs)
        batch_size = 64
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            if len(batch) == 0: continue
            centers = torch.tensor([p[0] for p in batch], dtype=torch.long)
            contexts = torch.tensor([p[1] for p in batch], dtype=torch.long)
            negatives = sampler(len(batch))
            loss = model(centers, contexts, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch == epochs - 1:
            return total_loss / len(pairs)

# 4. Lancer Optuna pour optimiser les hyperparamètres
if __name__ == "__main__":
    print("Optimisation en cours avec Optuna...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Meilleurs hyperparamètres :")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    best_params = study.best_params
    walks = generate_walks(G, num_walks=10, walk_length=best_params['walk_length'])
    pairs = generate_skipgram_pairs(walks, window_size=best_params['window_size'])
    sampler = negative_sampling(G, len(G.nodes), best_params['num_negatives'])

    model = SkipGramModel(len(G.nodes), best_params['embedding_dim'])
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])

    print("Reentraînement du meilleur modèle...")
    loss_history = []
    with open("loss_values.txt", "w") as lossfile:
        for epoch in range(20):
            total_loss = 0
            random.shuffle(pairs)
            for i in range(0, len(pairs), 64):
                batch = pairs[i:i+64]
                if len(batch) == 0: continue
                centers = torch.tensor([p[0] for p in batch], dtype=torch.long)
                contexts = torch.tensor([p[1] for p in batch], dtype=torch.long)
                negatives = sampler(len(batch))
                loss = model(centers, contexts, negatives)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(pairs)
            log_line = f"Epoch {epoch+1}/20 - Loss: {avg_loss:.4f}"
            print(log_line)
            lossfile.write(log_line + "\n")
            loss_history.append(avg_loss)

    torch.save(model.get_embeddings(), "embeddings_optimized.pt")
    print("Embeddings finaux sauvegardés dans 'embeddings_optimized.pt'")

