

# ### workimg
# import networkx as nx
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# from tqdm import tqdm

# class Node2Vec(nn.Module):
#     def __init__(self, graph, dimensions, walk_length, num_walks, p, q, T, learning_rate, window_size):
#         super(Node2Vec, self).__init__()
#         self.graph = graph
#         self.dimensions = dimensions
#         self.walk_length = walk_length
#         self.num_walks = num_walks
#         self.p = p
#         self.q = q
#         self.T = T  # Length of random walks for similarity
#         self.learning_rate = learning_rate
#         self.window_size = window_size

#         # Check if GPU is available and set device accordingly
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Initialize node embeddings on the selected device
#         self.node_embeddings = nn.Embedding(len(graph.nodes()), dimensions).to(self.device)
#         nn.init.xavier_uniform_(self.node_embeddings.weight)

#     def node2vec_walk(self, start_node):
#         walk = [start_node]
#         while len(walk) < self.walk_length:
#             current_node = walk[-1]
#             neighbors = list(self.graph.neighbors(current_node))
#             if len(neighbors) > 0:
#                 next_node = self.node2vec_step(current_node, walk[-2] if len(walk) > 1 else None)
#                 walk.append(next_node)
#             else:
#                 break
#         return walk

#     def node2vec_step(self, current_node, previous_node):
#         neighbors = list(self.graph.neighbors(current_node))
#         if previous_node is None:
#             return random.choice(neighbors)

#         weights = []
#         for neighbor in neighbors:
#             if neighbor == previous_node:
#                 weights.append(1 / self.p)
#             elif neighbor in self.graph[previous_node]:
#                 weights.append(1)
#             else:
#                 weights.append(1 / self.q)

#         normalized_weights = [w / sum(weights) for w in weights]
#         return random.choices(neighbors, weights=normalized_weights)[0]

#     def train(self):
#         optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
#         criterion = nn.CrossEntropyLoss()

#         # Use tqdm to create a progress bar with the total number of walks
#         pbar = tqdm(total=self.num_walks, desc="Total Walks")
#         for _ in range(self.num_walks):
#             pbar.update(1)  # Update the progress bar by 1 walk
#             for node in self.graph.nodes():
#                 walk = self.node2vec_walk(node)
#                 self.update_embeddings(walk, optimizer, criterion)
#         pbar.close()  # Close the progress bar when done

#     def update_embeddings(self, walk, optimizer, criterion):
#         for i, node in enumerate(walk):
#             for j in range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)):
#                 if i != j:
#                     node_i = torch.tensor(walk[i]).to(self.device)
#                     node_j = torch.tensor(walk[j]).to(self.device)

#                     # Calculate loss and update embeddings based on probability of visit
#                     loss = self.node_similarity_loss(node_i, node_j)
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()

#     def node_similarity_loss(self, node_i, node_j):
#         emb_i = self.node_embeddings(node_i)
#         emb_j = self.node_embeddings(node_j)

#         # Probability of visiting node_j on a length-T random walk starting at node_i
#         prob_ij = torch.sigmoid(torch.matmul(emb_i, emb_j.t()))

#         # small epsilon to avoid division by zero
#         epsilon = 1e-8
#         prob_ij = torch.max(prob_ij, torch.tensor(epsilon).to(self.device))

#         # Negative log likelihood as the loss
#         return -torch.log(prob_ij)

#     def get_embeddings(self):
#         return {node: self.node_embeddings(torch.tensor(node).to(self.device)).cpu().detach().numpy() for node in self.graph.nodes()}


##### also working
# import networkx as nx
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# class Node2Vec(nn.Module):
#     def __init__(self, graph, dimensions, walk_length, num_walks, p, q, learning_rate, window_size, epochs, negative_samples):
#         super(Node2Vec, self).__init__()
#         self.graph = graph
#         self.dimensions = dimensions
#         self.walk_length = walk_length
#         self.num_walks = num_walks
#         self.p = p
#         self.q = q
#         self.learning_rate = learning_rate
#         self.window_size = window_size
#         self.epochs = epochs
#         self.negative_samples = negative_samples

#         # Check if GPU is available and set device accordingly
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Initialize node embeddings on the selected device
#         self.node_embeddings = nn.Embedding(len(graph.nodes()), dimensions).to(self.device)
#         nn.init.xavier_uniform_(self.node_embeddings.weight)

#     def node2vec_walk(self, start_node):
#         walk = [start_node]
#         while len(walk) < self.walk_length:
#             current_node = walk[-1]
#             neighbors = list(self.graph.neighbors(current_node))
#             if len(neighbors) > 0:
#                 next_node = self.node2vec_step(current_node, walk[-2] if len(walk) > 1 else None)
#                 walk.append(next_node)
#             else:
#                 break
#         return walk

#     def node2vec_step(self, current_node, previous_node):
#         neighbors = list(self.graph.neighbors(current_node))
#         if previous_node is None:
#             return random.choice(neighbors)

#         weights = []
#         for neighbor in neighbors:
#             if neighbor == previous_node:
#                 weights.append(1 / self.p)
#             elif neighbor in self.graph[previous_node]:
#                 weights.append(1)
#             else:
#                 weights.append(1 / self.q)

#         normalized_weights = [w / sum(weights) for w in weights]
#         return random.choices(neighbors, weights=normalized_weights)[0]

#     def train(self):
#         optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
#         skipgram_losses = []

#         for epoch in range(self.epochs):
#             pbar = tqdm(total=self.num_walks, desc=f"Epoch {epoch + 1}/{self.epochs}")
#             total_loss = 0.0

#             for _ in range(self.num_walks):
#                 pbar.update(1)
#                 for node in self.graph.nodes():
#                     walk = self.node2vec_walk(node)
#                     loss = self.update_embeddings(walk, optimizer)
#                     total_loss += loss

#             pbar.close()
#             skipgram_losses.append(total_loss)

#         # Plot skip-gram loss
#         self.plot_loss(skipgram_losses)

#     def update_embeddings(self, walk, optimizer):
#         losses = []
#         for i, node in enumerate(walk):
#             for j in range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)):
#                 if i != j:
#                     node_i = torch.tensor(walk[i]).to(self.device)
#                     node_j = torch.tensor(walk[j]).to(self.device)

#                     # Calculate loss and update embeddings with Negative Sampling
#                     loss = self.node_similarity_loss(node_i, node_j)
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#                     losses.append(loss.item())

#         return sum(losses)

#     def node_similarity_loss(self, node_i, node_j):
#         emb_i = self.node_embeddings(node_i)
#         emb_j = self.node_embeddings(node_j)

#         # Positive example: Probability of visiting node_j from node_i
#         pos_score = torch.sigmoid(torch.sum(emb_i * emb_j, dim=-1))

#         # Negative examples: Sample negative nodes and calculate their scores
#         neg_samples = torch.randint(0, len(self.graph.nodes()), (self.negative_samples,)).to(self.device)
#         neg_emb = self.node_embeddings(neg_samples)
#         neg_score = torch.sigmoid(torch.sum(emb_i * neg_emb, dim=-1))

#         # Calculate Negative Sampling loss
#         loss = -torch.log(pos_score) - torch.sum(torch.log(1 - neg_score))
#         return loss

#     def get_embeddings(self):
#         return {node: self.node_embeddings(torch.tensor(node).to(self.device)).cpu().detach().numpy() for node in self.graph.nodes()}

#     def plot_loss(self, losses):
#         plt.figure(figsize=(10, 6))
#         plt.plot(range(1, self.epochs + 1), losses, marker='o', linestyle='-', color='b')
#         plt.title('Negative Sampling Loss Over Epochs')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.grid(True)
#         plt.show()






# ## working 
# import networkx as nx
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# class Node2Vec(nn.Module):
#     def __init__(self, graph, dimensions, walk_length, num_walks, p, q, learning_rate, window_size, epochs, negative_samples):
#         super(Node2Vec, self).__init__()
#         self.graph = graph
#         self.dimensions = dimensions
#         self.walk_length = walk_length
#         self.num_walks = num_walks
#         self.p = p
#         self.q = q
#         self.learning_rate = learning_rate
#         self.window_size = window_size
#         self.epochs = epochs
#         self.negative_samples = negative_samples

#         # Check if GPU is available and set device accordingly
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Initialize node embeddings on the selected device
#         self.node_embeddings = nn.Embedding(len(graph.nodes()), dimensions).to(self.device)
#         nn.init.xavier_uniform_(self.node_embeddings.weight)

#     def node2vec_walk(self, start_node):
#         walk = [start_node]
#         while len(walk) < self.walk_length:
#             current_node = walk[-1]
#             neighbors = list(self.graph.neighbors(current_node))
#             if len(neighbors) > 0:
#                 next_node = self.node2vec_step(current_node, walk[-2] if len(walk) > 1 else None)
#                 walk.append(next_node)
#             else:
#                 break
#         return walk

#     def node2vec_step(self, current_node, previous_node):
#         neighbors = list(self.graph.neighbors(current_node))
#         if previous_node is None:
#             return random.choice(neighbors)

#         weights = []
#         for neighbor in neighbors:
#             if neighbor == previous_node:
#                 weights.append(1 / self.p)
#             elif neighbor in self.graph[previous_node]:
#                 weights.append(1)
#             else:
#                 weights.append(1 / self.q)

#         normalized_weights = [w / sum(weights) for w in weights]
#         return random.choices(neighbors, weights=normalized_weights)[0]

#     def train(self):
#         optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
#         skipgram_losses = []

#         for epoch in range(self.epochs):
#             pbar = tqdm(total=self.num_walks, desc=f"Epoch {epoch + 1}/{self.epochs}")
#             total_loss = 0.0

#             for _ in range(self.num_walks):
#                 pbar.update(1)
#                 for node in self.graph.nodes():
#                     walk = self.node2vec_walk(node)
#                     loss = self.update_embeddings(walk, optimizer)
#                     total_loss += loss

#             pbar.close()
#             skipgram_losses.append(total_loss)

#         # Plot skip-gram loss
#         self.plot_loss(skipgram_losses)

#     def update_embeddings(self, walk, optimizer):
#         losses = []
#         for i, node in enumerate(walk):
#             for j in range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)):
#                 if i != j:
#                     node_i = torch.tensor(walk[i]).to(self.device)
#                     node_j = torch.tensor(walk[j]).to(self.device)

#                     # Calculate loss and update embeddings with Negative Sampling
#                     loss = self.node_similarity_loss(node_i, node_j)
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#                     losses.append(loss.item())

#         return sum(losses)

#     def node_similarity_loss(self, node_i, node_j):
#         emb_i = self.node_embeddings(node_i)
#         emb_j = self.node_embeddings(node_j)

#         # Positive example: Probability of visiting node_j from node_i
#         pos_score = torch.sigmoid(torch.sum(emb_i * emb_j, dim=-1))

#         # Negative examples: Sample negative nodes and calculate their scores
#         neg_samples = torch.randint(0, len(self.graph.nodes()), (self.negative_samples,)).to(self.device)
#         neg_emb = self.node_embeddings(neg_samples)
#         neg_score = torch.sigmoid(torch.sum(emb_i * neg_emb, dim=-1))

#         # Calculate Negative Sampling loss
#         loss = -torch.log(pos_score) - torch.sum(torch.log(1 - neg_score))
#         return loss

#     def get_embeddings(self):
#         return {node: self.node_embeddings(torch.tensor(node).to(self.device)).cpu().detach().numpy() for node in self.graph.nodes()}

#     def plot_loss(self, losses):
#         plt.figure(figsize=(10, 6))
#         plt.plot(range(1, self.epochs + 1), losses, marker='o', linestyle='-', color='b')
#         plt.title('Negative Sampling Loss Over Epochs')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.grid(True)
#         plt.show()

#     def compute_similarity_matrix(self):
#         num_nodes = len(self.graph.nodes())
#         similarity_matrix = np.zeros((num_nodes, num_nodes))

#         # Calculate pairwise similarity measures for all nodes
#         for i in range(num_nodes):
#             for j in range(i, num_nodes):
#                 node_i = i
#                 node_j = j

#                 # Calculate the probability of visiting node j on a random walk with the same length as walk_length starting at node i
#                 prob_ij = self.calculate_visit_probability(node_i, node_j)

#                 similarity_matrix[i, j] = prob_ij
#                 similarity_matrix[j, i] = prob_ij

#         return similarity_matrix

#     def calculate_visit_probability(self, node_i, node_j):
#         prob_ij = 0.0
#         for _ in range(self.walk_length):
#             walk = self.node2vec_walk(node_i)
#             if node_j in walk:
#                 prob_ij += 1.0

#         prob_ij /= self.walk_length

#         return prob_ij


# import networkx as nx
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# class Node2Vec(nn.Module):
#     def __init__(self, graph, dimensions, walk_length, num_walks, p, q, learning_rate, window_size, epochs, negative_samples):
#         super(Node2Vec, self).__init__()
#         self.graph = graph
#         self.dimensions = dimensions
#         self.walk_length = walk_length
#         self.num_walks = num_walks
#         self.p = p
#         self.q = q
#         self.learning_rate = learning_rate
#         self.window_size = window_size
#         self.epochs = epochs
#         self.negative_samples = negative_samples

#         # Check if GPU is available and set device accordingly
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Initialize node embeddings on the selected device
#         self.node_embeddings = nn.Embedding(len(graph.nodes()), dimensions).to(self.device)
#         nn.init.xavier_uniform_(self.node_embeddings.weight)

#     def node2vec_walk(self, start_node):
#         walk = [start_node]
#         while len(walk) < self.walk_length:
#             current_node = walk[-1]
#             neighbors = list(self.graph.neighbors(current_node))
#             if len(neighbors) > 0:
#                 next_node = self.node2vec_step(current_node, walk[-2] if len(walk) > 1 else None)
#                 walk.append(next_node)
#             else:
#                 break
#         return walk

#     def node2vec_step(self, current_node, previous_node):
#         neighbors = list(self.graph.neighbors(current_node))
#         if previous_node is None:
#             return random.choice(neighbors)

#         weights = []
#         for neighbor in neighbors:
#             if neighbor == previous_node:
#                 weights.append(1 / self.p)
#             elif neighbor in self.graph[previous_node]:
#                 weights.append(1)
#             else:
#                 weights.append(1 / self.q)

#         normalized_weights = [w / sum(weights) for w in weights]
#         return random.choices(neighbors, weights=normalized_weights)[0]

#     def train(self):
#         optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
#         skipgram_losses = []

#         for epoch in range(self.epochs):
#             pbar = tqdm(total=self.num_walks, desc=f"Epoch {epoch + 1}/{self.epochs}")
#             total_loss = 0.0

#             for _ in range(self.num_walks):
#                 pbar.update(1)
#                 for node in self.graph.nodes():
#                     walk = self.node2vec_walk(node)
#                     loss = self.update_embeddings(walk, optimizer)
#                     total_loss += loss

#             pbar.close()
#             skipgram_losses.append(total_loss)

#         # Plot skip-gram loss
#         self.plot_loss(skipgram_losses)

#     def update_embeddings(self, walk, optimizer):
#         losses = []
#         for i, node in enumerate(walk):
#             for j in range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)):
#                 if i != j:
#                     node_i = torch.tensor(walk[i]).to(self.device)
#                     node_j = torch.tensor(walk[j]).to(self.device)

#                     # Calculate loss and update embeddings with Negative Sampling
#                     loss = self.node_similarity_loss(node_i, node_j)
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#                     losses.append(loss.item())

#         return sum(losses)

#     def node_similarity_loss(self, node_i, node_j):
#         emb_i = self.node_embeddings(node_i)
#         emb_j = self.node_embeddings(node_j)

#         # Positive example: Probability of visiting node_j from node_i
#         pos_score = torch.sigmoid(torch.sum(emb_i * emb_j, dim=-1))

#         # Negative examples: Sample negative nodes and calculate their scores
#         neg_samples = torch.randint(0, len(self.graph.nodes()), (self.negative_samples,)).to(self.device)
#         neg_emb = self.node_embeddings(neg_samples)
#         neg_score = torch.sigmoid(torch.sum(emb_i * neg_emb, dim=-1))

#         # Calculate Negative Sampling loss
#         loss = -torch.log(pos_score) - torch.sum(torch.log(1 - neg_score))
#         return loss

#     def get_embeddings(self):
#         # Transpose the embeddings to have shape (dimensions, number of nodes in the graph)
#         embeddings = self.node_embeddings.weight.data.T.cpu().detach().numpy()
#         return embeddings

#     def plot_loss(self, losses):
#         plt.figure(figsize=(10, 6))
#         plt.plot(range(1, self.epochs + 1), losses, marker='o', linestyle='-', color='b')
#         plt.title('Negative Sampling Loss Over Epochs')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.grid(True)
#         plt.show()

#     def compute_similarity_matrix(self):
#         num_nodes = len(self.graph.nodes())
#         similarity_matrix = np.zeros((num_nodes, num_nodes))

#         # Calculate pairwise similarity measures for all nodes
#         for i in range(num_nodes):
#             for j in range(i, num_nodes):
#                 node_i = i
#                 node_j = j

#                 # Calculate the probability of visiting node j on a random walk with the same length as walk_length starting at node i
#                 prob_ij = self.calculate_visit_probability(node_i, node_j)

#                 similarity_matrix[i, j] = prob_ij
#                 similarity_matrix[j, i] = prob_ij

#         return similarity_matrix

#     def calculate_visit_probability(self, node_i, node_j):
#         prob_ij = 0.0
#         for _ in range(self.walk_length):
#             walk = self.node2vec_walk(node_i)
#             if node_j in walk:
#                 prob_ij += 1.0

#         prob_ij /= self.walk_length

#         return prob_ij


import os
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Node2Vec(nn.Module):
    def __init__(self, graph, dimensions, walk_length, num_walks, p, q, learning_rate, window_size, epochs, negative_samples):
        super(Node2Vec, self).__init__()
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.epochs = epochs
        self.negative_samples = negative_samples

        # Check if GPU is available and set device accordingly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.device = "cpu"

        # Initialize node embeddings on the selected device
        self.node_embeddings = nn.Embedding(len(graph.nodes()), dimensions).to(self.device)
        nn.init.xavier_uniform_(self.node_embeddings.weight)

    def node2vec_walk(self, start_node):
        walk = [start_node]
        while len(walk) < self.walk_length:
            current_node = walk[-1]
            neighbors = list(self.graph.neighbors(current_node))
            if len(neighbors) > 0:
                next_node = self.node2vec_step(current_node, walk[-2] if len(walk) > 1 else None)
                walk.append(next_node)
            else:
                break
        return walk

    def node2vec_step(self, current_node, previous_node):
        neighbors = list(self.graph.neighbors(current_node))
        if previous_node is None:
            return random.choice(neighbors)

        weights = []
        for neighbor in neighbors:
            if neighbor == previous_node:
                weights.append(1 / self.p)
            elif neighbor in self.graph[previous_node]:
                weights.append(1)
            else:
                weights.append(1 / self.q)

        normalized_weights = [w / sum(weights) for w in weights]
        return random.choices(neighbors, weights=normalized_weights)[0]

#     def train(self, batch_size):
#         optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
#         skipgram_losses = []

#         for epoch in range(self.epochs):
#             pbar = tqdm(total=self.num_walks, desc=f"Epoch {epoch + 1}/{self.epochs}")
#             total_loss = 0.0

#             walks = []
#             for _ in range(self.num_walks):
#                 for node in self.graph.nodes():
#                     walk = self.node2vec_walk(node)
#                     walks.append(walk)

#             for batch_start in range(0, len(walks), batch_size):
#                 batch_end = min(batch_start + batch_size, len(walks))
#                 batch_walks = walks[batch_start:batch_end]
#                 loss = self.update_embeddings(batch_walks, optimizer)
#                 total_loss += loss

#             pbar.close()
#             skipgram_losses.append(total_loss)

#         # Plot skip-gram loss
#         self.plot_loss(skipgram_losses)

    def train(self, batch_size):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)  # Create an Adam optimizer
        skipgram_losses = []

        for epoch in range(self.epochs):
            pbar = tqdm(total=self.num_walks, desc=f"Epoch {epoch + 1}/{self.epochs}")
            total_loss = 0.0

            walks = []
            for _ in range(self.num_walks):
                for node in self.graph.nodes():
                    walk = self.node2vec_walk(node)
                    walks.append(walk)

            for batch_start in range(0, len(walks), batch_size):
                batch_end = min(batch_start + batch_size, len(walks))
                batch_walks = walks[batch_start:batch_end]
                loss = self.update_embeddings(batch_walks, optimizer)
                total_loss += loss

            pbar.close()
            skipgram_losses.append(total_loss)

        # Plot skip-gram loss
        self.plot_loss(skipgram_losses)

    def update_embeddings(self, walks, optimizer):
        losses = []
        for walk in walks:
            for i, node in enumerate(walk):
                for j in range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)):
                    if i != j:
                        node_i = torch.tensor(walk[i]).to(self.device)
                        node_j = torch.tensor(walk[j]).to(self.device)

                        # Calculate loss and update embeddings with Negative Sampling
                        loss = self.node_similarity_loss(node_i, node_j)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        losses.append(loss.item())

        return sum(losses)

    def node_similarity_loss(self, node_i, node_j):
        emb_i = self.node_embeddings(node_i)
        emb_j = self.node_embeddings(node_j)

        # Positive example: Probability of visiting node_j from node_i
        pos_score = torch.sigmoid(torch.sum(emb_i * emb_j, dim=-1))

        # Negative examples: Sample negative nodes and calculate their scores
        neg_samples = torch.randint(0, len(self.graph.nodes()), (self.negative_samples,)).to(self.device)
        neg_emb = self.node_embeddings(neg_samples)
        neg_score = torch.sigmoid(torch.sum(emb_i * neg_emb, dim=-1))

        # Calculate Negative Sampling loss
        loss = -torch.log(pos_score) - torch.sum(torch.log(1 - neg_score))
        return loss

    def get_embeddings(self):
        # Transpose the embeddings to have shape (dimensions, number of nodes in the graph)
        embeddings = self.node_embeddings.weight.data.T.cpu().detach().numpy()
        return embeddings

#     def plot_loss(self, losses):
#         plt.figure(figsize=(10, 6))
#         plt.plot(range(1, self.epochs + 1), losses, marker='o', linestyle='-', color='b')
#         plt.title('Negative Sampling Loss Over Epochs')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.grid(True)
#         plt.show()

    def plot_loss(self, losses):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.epochs + 1), losses, marker='o', linestyle='-', color='b')
        plt.title('Negative Sampling Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        # Generate a filename based on parameters and datetime
        filename = self.generate_filename()

        # Create a directory if it doesn't exist
        os.makedirs("PLOT_LOSS", exist_ok=True)

        # Save the plot in the "PLOT LOSS" folder with the generated filename
        plt.savefig(os.path.join("PLOT_LOSS", filename))
        plt.show()

    def generate_filename(self):
        # Generate a filename based on parameters and datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"params_dim{self.dimensions}_len{self.walk_length}_walks{self.num_walks}_p{self.p}_q{self.q}_lr{self.learning_rate}_ws{self.window_size}_epochs{self.epochs}_neg{self.negative_samples}_{timestamp}.png"
        return filename

    def compute_similarity_matrix(self):
        num_nodes = len(self.graph.nodes())
        similarity_matrix = np.zeros((num_nodes, num_nodes))

        # Calculate pairwise similarity measures for all nodes
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                node_i = i
                node_j = j

                # Calculate the probability of visiting node j on a random walk with the same length as walk_length starting at node i
                prob_ij = self.calculate_visit_probability(node_i, node_j)

                similarity_matrix[i, j] = prob_ij
                similarity_matrix[j, i] = prob_ij

        return similarity_matrix

    def calculate_visit_probability(self, node_i, node_j):
        prob_ij = 0.0
        for _ in range(self.walk_length):
            walk = self.node2vec_walk(node_i)
            if node_j in walk:
                prob_ij += 1.0

        prob_ij /= self.walk_length

        return prob_ij