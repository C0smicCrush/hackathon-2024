import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as ptud
import torch_geometric as ptg
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import os
import time
import random
import json
import sentence_transformers
import csv
from tqdm.auto import tqdm
import pandas as pd
from numpy.linalg import norm
from sklearn.cluster import KMeans
from fakerFile import generate_fake_response, required_questions, optional_questions, generate_fake_profile
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

mySentenceTransformer = sentence_transformers.SentenceTransformer('paraphrase-MiniLM-L6-v2')

possiblePeople = []

for i in tqdm(range(150)):
    fakerResponses = generate_fake_profile()
    possiblePeople.append(fakerResponses)

with open('possiblePeople2.json', 'w') as file:
    json.dump(possiblePeople, file, indent=4)

# #possible people to json
with open('possiblePeople2.json', 'w') as file:
    json.dump(possiblePeople, file)


# # Load possible people from json
with open('possiblePeople2.json', 'r') as file:
    possiblePeople = json.load(file)
embeddings = mySentenceTransformer.encode(possiblePeople)
print(embeddings)
edgelist1 = []
edgelist2 = []
minin = 10
sum = 0
for i in range(150):
    for j in range(150):
        if i != j and np.dot(embeddings[i], embeddings[j])/norm(embeddings[i])/norm(embeddings[j]) > 0.7:
            edgelist1.append(i)
            edgelist2.append(j)
with open('possiblePeople2.json', 'r') as file:
    possiblePeople = json.load(file)
#print(sum/(50*49))
#print(minin)




x = torch.tensor(embeddings, dtype=torch.float)


edge_index = torch.tensor([[0, 1, 2, 3],
                            [1, 2, 3, 0]], dtype=torch.long)


edge_index = torch.tensor([edgelist1, edgelist2], dtype=torch.long)

data = Data(x=x, edge_index=edge_index)

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(384, 48)  
        self.conv2 = GCNConv(48, 384)  

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in (range(200)):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data)
    loss = F.mse_loss(out, data.x)  # You may want to use a different method for your embeddings
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')



# Get the node embeddings
model.eval()
embeddings = model(data).detach().numpy()

# Perform K-Means clustering
kmeans = KMeans(n_clusters=12)  # Choose number of clusters
predicted_labels = kmeans.fit_predict(embeddings)

print("Predicted Clusters:", predicted_labels)
for i in range(12):
    print("Cluster", i, ":", [possiblePeople[j] for j in range(50) if predicted_labels[j] == i])


df = pd.DataFrame(possiblePeople)


df['What is your budget?'] = df['What is your budget?'].str.replace('$', '').str.replace(' per month', '').astype(float)
df['What time do you usually go to bed?'] = df['What time do you usually go to bed?'].str.replace(' PM', '').astype(int)

def convert_lease_length(length):
    length = length.lower()
    if length == '1 year' :
        return 12
    out = int(length.split()[0])
    return out

features = df[['What is your age?', 'What is your budget?', 'What time do you usually go to bed?']].values
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)


similarity_matrix = cosine_similarity(features_normalized)
threshold = 0.5  
adjacency_matrix = (similarity_matrix > threshold).astype(int)
G = nx.from_numpy_array(adjacency_matrix)

centrality = nx.degree_centrality(G)
centrality_df = pd.DataFrame(list(centrality.items()), columns=['Node', 'Centrality'])
centrality_df['Profile'] = df['What is your name?']

# Find outliers: Here we define outliers as those with lower centrality
threshold = np.percentile(centrality_df['Centrality'], 5)  # You can adjust the threshold

outliers = centrality_df[centrality_df['Centrality'] < threshold]

# Output outliers
print("Identified Outliers:")
print(outliers[['Profile', 'Centrality']])