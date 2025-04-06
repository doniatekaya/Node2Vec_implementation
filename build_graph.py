# build_graph_sparse.py
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import pickle

# Fonction Haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1) * cos(phi2) * sin(dlambda/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

# Seuils plus stricts
DIST_MAX = 20     # Moins de 20m
TIME_MAX = 5 * 60 # 5 minutes

# Chargement des données
file = "node2vecgps_data_covid_sparse.csv"
df = pd.read_csv(file)

# Construction du graphe
G = nx.Graph()
for _, row in df.iterrows():
    G.add_node(row['id'], 
               infected=row['infected'], 
               timestamp=row['timestamp'], 
               latitude=row['latitude'], 
               longitude=row['longitude'])

nodes = list(G.nodes(data=True))
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        n1, d1 = nodes[i]
        n2, d2 = nodes[j]

        dist = haversine(d1['latitude'], d1['longitude'], d2['latitude'], d2['longitude'])
        time_diff = abs(d1['timestamp'] - d2['timestamp'])

        if dist <= DIST_MAX and time_diff <= TIME_MAX:
            G.add_edge(n1, n2)

# Sauvegarde
with open("graph.pkl", "wb") as f:
    pickle.dump(G, f)

print(f"✅ Graphe sauvegardé avec {G.number_of_nodes()} nœuds et {G.number_of_edges()} arêtes")

# Affichage du graphe
pos = nx.spring_layout(G, seed=42)
colors = ["red" if G.nodes[n]["infected"] else "green" for n in G.nodes]
plt.figure(figsize=(10, 7))
nx.draw(G, pos, with_labels=True, node_color=colors, edge_color="gray", node_size=300)
plt.title("Graphe de Contact COVID (Sparse)")
plt.savefig("graph_sparse.png")
plt.show()
