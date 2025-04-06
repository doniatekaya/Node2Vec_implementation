# generate_data_propagation.py
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import networkx as nx
from math import radians, sin, cos, sqrt, atan2

# Fonction pour calculer la distance Haversine (mètres)
"""calculer la distance entre deux points géographiques
 (latitudes/longitudes) en utilisant la formule de haversine qui tient compte de la courbure de la terre."""
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2*R*atan2(sqrt(a), sqrt(1 - a))

# Paramètres
n_individus = 80
n_points_par_personne = 3
base_time = datetime(2020, 5, 1, 12, 0, 0)
center_lat, center_lon = 48.8566, 2.3522

# Génération des données (sans infection)
data = []
for i in range(n_individus):
    person_id = f"pers_{i}"
    for _ in range(n_points_par_personne):
        lat = center_lat + np.random.uniform(-0.0003, 0.0003)
        lon = center_lon + np.random.uniform(-0.0003, 0.0003)
        timestamp = base_time + timedelta(minutes=random.randint(0, 20))
        timestamp_unix = int(timestamp.timestamp())
        data.append({
            "id": person_id,
            "timestamp": timestamp_unix,
            "latitude": lat,
            "longitude": lon,
            "infected": False
        })

df = pd.DataFrame(data)

# Construire un graphe temporaire pour propager l'infection
G = nx.Graph()
for i, row in df.iterrows():
    G.add_node(row['id'], **row.to_dict())

nodes = list(G.nodes(data=True))
DIST_MAX = 20
TIME_MAX = 5 * 60

for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        n1, d1 = nodes[i]
        n2, d2 = nodes[j]
        dist = haversine(d1['latitude'], d1['longitude'], d2['latitude'], d2['longitude'])
        time_diff = abs(d1['timestamp'] - d2['timestamp'])
        if dist <= DIST_MAX and time_diff <= TIME_MAX:
            G.add_edge(n1, n2)

# Propagation de l'infection
sources = random.sample(list(G.nodes), 3)
for src in sources:
    G.nodes[src]['infected'] = True
    for neighbor in nx.single_source_shortest_path_length(G, src, cutoff=1):
        if random.random() < 0.5:
            G.nodes[neighbor]['infected'] = True

# Réinjecter dans DataFrame
for i, (node, data_dict) in enumerate(G.nodes(data=True)):
    df.loc[df['id'] == node, 'infected'] = data_dict['infected']

# Sauvegarde finale
output_file = "node2vecgps_data_covid_sparse.csv"
df.to_csv(output_file, index=False)
print(f"Données simulées avec propagation sauvegardées dans {output_file}")




