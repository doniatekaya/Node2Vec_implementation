import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import networkx as nx

class SkipGramModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(SkipGramModel, self).__init__()
        """un pour le role de centre (u_embeddings)
        un autre pour le role de contexte (v_embeddings)."""
        self.u_embeddings = nn.Embedding(num_nodes, embedding_dim)  
        self.v_embeddings = nn.Embedding(num_nodes, embedding_dim)

        # initialisation des poids
        init_range = 0.5 / embedding_dim
        self.u_embeddings.weight.data.uniform_(-init_range, init_range)   #initialiser u_embeddings aleatoirement
        self.v_embeddings.weight.data.uniform_(-0, 0)  #initialiser v_embeddings à zero

    def forward(self, center_nodes, context_nodes, negative_nodes):
        emb_center = self.u_embeddings(center_nodes)
        emb_context = self.v_embeddings(context_nodes)
        emb_negative = self.v_embeddings(negative_nodes)

        # Produit scalaire positif
        pos_score = torch.sum(emb_center * emb_context, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        # Produit scalaire négatif
        neg_score = torch.bmm(emb_negative, emb_center.unsqueeze(2)).squeeze()
        neg_loss = F.logsigmoid(-neg_score).sum(1)

        return -(pos_loss + neg_loss).mean()

    def get_embeddings(self):
        return self.u_embeddings.weight.data.cpu().numpy()

"""Generation des random walks"""
def generate_walks(graph, num_walks=10, walk_length=20):
    walks = []
    nodes = list(graph.nodes())
    """Simule des promenades aléatoires sur le graphe. Chaque nœud génère num_walks parcours de longueur walk_length."""
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                neighbors = list(graph.neighbors(walk[-1]))
                if neighbors:
                    walk.append(random.choice(neighbors))
                else:
                    break
            walks.append(walk)
    return walks


def generate_skipgram_pairs(walks, window_size=5):
    pairs = []
    for walk in walks:
        for i, center in enumerate(walk):
            context_range = range(max(0, i - window_size), min(len(walk), i + window_size + 1))
            for j in context_range:
                if i != j:
                    pairs.append((center, walk[j]))
    return pairs


def negative_sampling(graph, num_nodes, num_negatives=5):
    """
    Pour chaque edge (u,v), on génère `num_negatives` nœuds aléatoires comme négatifs.
    """
    all_nodes = list(graph.nodes())
    def sample_negatives(batch_size):
        return torch.randint(0, num_nodes, (batch_size, num_negatives))
    return sample_negatives

