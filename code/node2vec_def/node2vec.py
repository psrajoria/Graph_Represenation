# # # import numpy as np
# # # import networkx as nx
# # # from gensim.models import Word2Vec
# # # import matplotlib.pyplot as plt

# # # class Node2VecOptimized:
# # #     """
# # #     Node2VecOptimized: A class for generating embeddings of nodes in a graph using Node2Vec.

# # #     Args:
# # #         graph (nx.Graph): The input graph for which embeddings will be generated.
# # #         dimensions (int): The dimensionality of the node embeddings.
# # #         num_walks (int): The number of random walks to perform for each node.
# # #         walk_length (int): The length of each random walk.
# # #         window_size (int): The context window size for Word2Vec.
# # #         num_epochs (int): The number of training epochs.
# # #         p (float): Return parameter for biased random walks.
# # #         q (float): In-out parameter for biased random walks.
# # #         learning_rate (float): Learning rate for gradient descent.
# # #         regularization_coeff (float): Coefficient for L2 regularization.
# # #         workers (int): The number of parallel workers for generating random walks.

# # #     Attributes:
# # #         graph (nx.Graph): The input graph.
# # #         dimensions (int): The dimensionality of the node embeddings.
# # #         num_walks (int): The number of random walks to perform for each node.
# # #         walk_length (int): The length of each random walk.
# # #         window_size (int): The context window size for Word2Vec.
# # #         num_epochs (int): The number of training epochs.
# # #         p (float): Return parameter for biased random walks.
# # #         q (float): In-out parameter for biased random walks.
# # #         learning_rate (float): Learning rate for gradient descent.
# # #         regularization_coeff (float): Coefficient for L2 regularization.
# # #         workers (int): The number of parallel workers for generating random walks.
# # #         walks (list of list): List of random walks generated for the graph.
# # #         embeddings (numpy.ndarray): The node embeddings.

# # #     Methods:
# # #         generate_walks(): Generate random walks on the graph.
# # #         weighted_choice(neighbors, prev_node, curr_node): Choose the next node for a random walk.
# # #         initialize_embeddings(): Initialize node embeddings randomly.
# # #         train_embedding(): Train the embeddings using the Node2Vec algorithm.
# # #         compute_similarity_matrix(): Compute the similarity matrix for the graph.
# # #         plot_loss(loss_history): Plot the loss during training.
# # #         create_word2vec_embeddings(dimensions, window, sg, epochs): Create Word2Vec embeddings.

# # #     Example usage:
# # #     ```python
# # #     graph = nx.Graph()  # Create your graph here
# # #     node2vec = Node2VecOptimized(graph)
# # #     embeddings = node2vec.embed_all_nodes()
# # #     node2vec.plot_loss(node2vec.loss_history)

# # #     # Create Word2Vec embeddings
# # #     word2vec_embeddings = node2vec.create_word2vec_embeddings(dimensions=64, window=10, sg=1, epochs=10)
# # #     node2vec.plot_loss(node2vec.train_embedding())  # Plot the Node2Vec loss
# # #     node2vec.plot_loss(node2vec.create_word2vec_embeddings(dimensions=64, window=10, sg=1, epochs=10))  # Plot the Word2Vec loss
# # #     ```
# # #     """

# # #     def __init__(self, graph, dimensions=64, num_walks=10, walk_length=80,
# # #                  window_size=10, num_epochs=100, p=1.0, q=1.0,
# # #                  learning_rate=0.025, regularization_coeff=0.1, workers=4):
# # #         # Initialize class attributes
# # #         self.graph = graph
# # #         self.dimensions = dimensions
# # #         self.num_walks = num_walks
# # #         self.walk_length = walk_length
# # #         self.window_size = window_size
# # #         self.num_epochs = num_epochs
# # #         self.p = p
# # #         self.q = q
# # #         self.learning_rate = learning_rate
# # #         self.regularization_coeff = regularization_coeff
# # #         self.workers = workers
# # #         self.walks = self.generate_walks()  # Generate random walks
# # #         self.embeddings = self.initialize_embeddings()  # Initialize node embeddings
# # #         self.loss_history = []  # To store loss during training


# # #     def generate_walks(self):
# # #         """
# # #         Generate random walks on the graph.

# # #         Returns:
# # #             list of list: List of random walks generated for the graph.
# # #         """
# # #         def node2vec_walk(start_node):
# # #             walk = [start_node]
# # #             while len(walk) < self.walk_length:
# # #                 curr_node = walk[-1]
# # #                 neighbors = list(self.graph.neighbors(curr_node))
# # #                 if len(neighbors) > 0:
# # #                     if len(walk) == 1:
# # #                         walk.append(np.random.choice(neighbors))
# # #                     else:
# # #                         prev_node = walk[-2]
# # #                         next_node = self.weighted_choice(neighbors, prev_node, curr_node)  # Compute transition probabilities
# # #                         walk.append(next_node)
# # #                 else:
# # #                     break
# # #             return walk

# # #         # Generate random walks for all nodes in parallel
# # #         walks = [node2vec_walk(node) for _ in range(self.num_walks) for node in self.graph.nodes()]
# # #         return walks

# # #     def weighted_choice(self, neighbors, prev_node, curr_node):
# # #         """
# # #         Choose the next node for a random walk based on weights.

# # #         Args:
# # #             neighbors (list): List of neighboring nodes.
# # #             prev_node (int): Previous node in the random walk.
# # #             curr_node (int): Current node in the random walk.

# # #         Returns:
# # #             int: The next node to visit in the random walk.
# # #         """
# # #         weights = [1.0 / self.p if neighbor == prev_node
# # #                    else 1.0 if self.graph.has_edge(curr_node, neighbor)
# # #                    else 1.0 / self.q for neighbor in neighbors]  # Transition probabilities
# # #         weights = np.array(weights)
# # #         weights /= weights.sum()
# # #         next_node = np.random.choice(neighbors, p=weights)
# # #         return next_node

# # #     def initialize_embeddings(self):
# # #         """
# # #         Initialize node embeddings randomly.

# # #         Returns:
# # #             numpy.ndarray: Randomly initialized node embeddings.
# # #         """
# # #         num_nodes = len(self.graph.nodes())
# # #         return np.random.rand(num_nodes, self.dimensions)

# # #     def train_embedding(self):
# # #         """
# # #         Train the embeddings using the Node2Vec algorithm.

# # #         Returns:
# # #             list: List of loss values during training.
# # #         """
# # #         loss_history = []
# # #         for epoch in range(self.num_epochs):
# # #             z_transpose = np.transpose(self.embeddings)
# # #             similarity_matrix = self.compute_similarity_matrix()
# # #             loss = np.linalg.norm(np.dot(z_transpose, self.embeddings) - similarity_matrix)
# # #             loss_history.append(loss)

# # #             gradient = -2 * np.dot(self.embeddings, similarity_matrix) + 2 * np.dot(self.embeddings, np.dot(z_transpose, self.embeddings))
# # #             gradient += 2 * self.regularization_coeff * self.embeddings

# # #             self.embeddings -= self.learning_rate * gradient

# # #             # Early stopping based on loss change
# # #             if epoch > 0 and abs(loss_history[-1] - loss_history[-2]) < 1e-5:
# # #                 break

# # #         self.loss_history = loss_history
# # #         return loss_history

# # #     def compute_similarity_matrix(self):
# # #         """
# # #         Compute the similarity matrix for the graph based on both structure and Euclidean distance.

# # #         Returns:
# # #             numpy.ndarray: The computed similarity matrix.
# # #         """
# # #         num_nodes = len(self.graph.nodes())
# # #         similarity_matrix = np.zeros((num_nodes, num_nodes))

# # #         # Compute structural similarity (Jaccard similarity)
# # #         for i in range(num_nodes):
# # #             for j in range(i, num_nodes):
# # #                 if i != j:
# # #                     neighbors_i = set(self.graph.neighbors(i))
# # #                     neighbors_j = set(self.graph.neighbors(j))
# # #                     jaccard_similarity = len(neighbors_i.intersection(neighbors_j)) / len(neighbors_i.union(neighbors_j))
# # #                     similarity_matrix[i, j] = jaccard_similarity
# # #                     similarity_matrix[j, i] = jaccard_similarity

# # #         # Compute Euclidean distance-based similarity
# # #         node_embeddings = self.embeddings
# # #         for i in range(num_nodes):
# # #             for j in range(i, num_nodes):
# # #                 if i != j:
# # #                     euclidean_distance = np.linalg.norm(node_embeddings[i] - node_embeddings[j])
# # #                     euclidean_similarity = 1 / (1 + euclidean_distance)  # Inverse of distance as similarity
# # #                     similarity_matrix[i, j] += euclidean_similarity
# # #                     similarity_matrix[j, i] += euclidean_similarity

# # #         return similarity_matrix
    
# # # #     def create_defined_embedding(self):
# # # #     """
# # # #         Create embeddings using the defined trained embedding with the computed similarity matrix.

# # # #         Returns:
# # # #         numpy.ndarray: Node embeddings using the defined trained embedding.
# # # #     """
# # # #         dimensions = self.dimensions  # Use specified dimensions
# # # #         similarity_matrix = self.compute_similarity_matrix()
# # # #         eig_values, eig_vectors = np.linalg.eigh(similarity_matrix)
# # # #         sorted_indices = np.argsort(eig_values)[::-1]
# # # #         eig_vectors = eig_vectors[:, sorted_indices]
# # # #         embeddings = eig_vectors[:, :dimensions]  # Use specified dimensions
# # # #         return embeddings

# # #     def create_defined_embedding(self):
# # #         """
# # #         Create embeddings using the defined trained embedding with the computed similarity matrix.

# # #         Returns:
# # #         numpy.ndarray: Node embeddings using the defined trained embedding.
# # #         """
# # #         # Train the embedding using the train_embedding method
# # #         self.train_embedding()

# # #         # Return the trained embeddings
# # #         return self.embeddings
    

    
# # #     def plot_loss(self, loss_history):
# # #         """
# # #         Plot the loss during training.

# # #         Args:
# # #             loss_history (list): List of loss values during training.
# # #         """
# # #         plt.figure(figsize=(10, 5))
# # #         plt.plot(loss_history, label="Loss")
# # #         plt.xlabel("Epoch")
# # #         plt.ylabel("Loss")
# # #         plt.legend()
# # #         plt.title("Loss During Training")
# # #         plt.show()

# # #     def create_word2vec_embeddings(self, dimensions=64, window=10, sg=1, epochs=10):
# # #         """
# # #         Create Word2Vec embeddings.

# # #         Args:
# # #             dimensions (int): The dimensionality of the Word2Vec embeddings.
# # #             window (int): The context window size for Word2Vec.
# # #             sg (int): Training algorithm for Word2Vec (skip-gram or CBOW).
# # #             epochs (int): Number of training epochs for Word2Vec.

# # #         Returns:
# # #             numpy.ndarray: Word2Vec embeddings for nodes.
# # #         """
# # #         sentences = [list(map(str, walk)) for walk in self.walks]
# # #         w2v_model = Word2Vec(sentences, vector_size=dimensions, window=window, sg=sg, epochs=epochs)
# # #         embeddings = np.array([w2v_model.wv[str(node)] for node in range(len(self.graph.nodes()))])
# # #         return embeddings

# # # # # Example usage:
# # # # graph = nx.Graph()  # Create your graph here
# # # # # Add nodes and edges to 'graph' as needed

# # # # node2vec = Node2VecOptimized(graph, dimensions=64)  # Specify dimensions here

# # # # # Create defined trained embedding
# # # # defined_embeddings = node2vec.create_defined_embedding()
# # # # node2vec.plot_loss(node2vec.train_embedding())  # Plot the Node2Vec loss

# # # # # Create Word2Vec embeddings
# # # # word2vec_embeddings = node2vec.create_word2vec_embeddings(dimensions=64, window=10, sg=1, epochs=10)
# # # # node2vec.plot_loss(node2vec.create_word2vec_embeddings(dimensions=64, window=10, sg=1, epochs=10))  # Plot the Word2Vec loss

# # import numpy as np
# # import networkx as nx
# # from tqdm import tqdm
# # import matplotlib.pyplot as plt
# # from sklearn.metrics.pairwise import pairwise_distances
# # from gensim.models import Word2Vec
# # from sklearn.decomposition import PCA
# # from sklearn.manifold import TSNE

# # class Node2Vec:
# #     def __init__(self, dimensions, num_walks, walk_length, num_epochs, p, q):
# #         self.dimensions = dimensions
# #         self.num_walks = num_walks
# #         self.walk_length = walk_length
# #         self.num_epochs = num_epochs
# #         self.p = p
# #         self.q = q
# #         self.node_embeddings = None
# #         self.word2vec_embeddings = None
# #         self.node2vec_loss_history = []
# #         self.word2vec_loss_history = []

# #     def generate_random_walks(self, graph):
# #         walks = []
# #         nodes = list(graph.nodes)

# #         for _ in tqdm(range(self.num_walks), desc="Generating Random Walks"):
# #             np.random.shuffle(nodes)
# #             for node in nodes:
# #                 walk = self.generate_single_random_walk(graph, node)
# #                 walks.append(walk)

# #         return walks

# #     def generate_single_random_walk(self, graph, start_node):
# #         walk = [start_node]
# #         current_node = start_node

# #         while len(walk) < self.walk_length:
# #             neighbors = list(graph.neighbors(current_node))

# #             if len(neighbors) == 0:
# #                 break

# #             next_node = self.choose_next_node(graph, current_node, neighbors)
# #             walk.append(next_node)
# #             current_node = next_node

# #         return walk

# # #     def choose_next_node(self, graph, current_node, neighbors):
# # #         if len(self.walk) == 1:
# # #             return np.random.choice(neighbors)
        
# # #         probabilities = []
# # #         for neighbor in neighbors:
# # #             if neighbor == self.walk[-2]:
# # #                 probabilities.append(1 / self.p)
# # #             elif neighbor in graph.neighbors(self.walk[-2]):
# # #                 probabilities.append(1)
# # #             else:
# # #                 probabilities.append(1 / self.q)

# # #         probabilities = np.array(probabilities)
# # #         probabilities /= probabilities.sum()
# # #         return np.random.choice(neighbors, p=probabilities)

# #     def choose_next_node(self, graph, current_node, neighbors):
# #         if len(self.walk) == 1:
# #             return np.random.choice(neighbors)

# #         probabilities = []
# #         for neighbor in neighbors:
# #             if neighbor == self.walk[-2]:
# #                 probabilities.append(1 / self.p)
# #             elif neighbor in graph.neighbors(self.walk[-2]):
# #                 probabilities.append(1)
# #             else:
# #                 probabilities.append(1 / self.q)

# #         probabilities = np.array(probabilities)
# #         probabilities /= probabilities.sum()
# #         return np.random.choice(neighbors, p=probabilities)

# #     def calculate_similarity_matrix(self, graph, walks):
# #         nodes = list(graph.nodes)
# #         num_nodes = len(nodes)
# #         similarity_matrix = np.zeros((num_nodes, num_nodes))

# #         neighbor_sets = {node: set(graph.neighbors(node)) for node in nodes}

# #         for i in tqdm(range(num_nodes), desc="Calculating Similarity Matrix"):
# #             for j in range(i + 1, num_nodes):
# #                 node1, node2 = nodes[i], nodes[j]
# #                 jaccard_similarity = self.calculate_jaccard_similarity(neighbor_sets, node1, node2)
# #                 embeddings1, embeddings2 = self.calculate_node_embeddings(walks, node1, node2)
# #                 euclidean_distance = np.linalg.norm(embeddings1 - embeddings2)

# #                 # Use Euclidean distance directly as similarity
# #                 similarity_matrix[i][j] = 1 / (1 + euclidean_distance)
# #                 similarity_matrix[j][i] = similarity_matrix[i][j]

# #         return similarity_matrix

# #     def calculate_jaccard_similarity(self, neighbor_sets, node1, node2):
# #         neighbors1 = neighbor_sets[node1]
# #         neighbors2 = neighbor_sets[node2]
# #         intersection = len(neighbors1.intersection(neighbors2))
# #         union = len(neighbors1.union(neighbors2))
# #         return intersection / union if union != 0 else 0

# #     def calculate_node_embeddings(self, walks, node1, node2):
# #         embeddings1, embeddings2 = np.zeros(self.dimensions), np.zeros(self.dimensions)

# #         for walk in walks:
# #             if node1 in walk:
# #                 embeddings1 += np.array(walk).count(node1) * np.array(walk)
# #             if node2 in walk:
# #                 embeddings2 += np.array(walk).count(node2) * np.array(walk)

# #         return embeddings1, embeddings2

# #     def calculate_loss(self, Z, S):
# #         loss = np.linalg.norm(np.dot(Z, Z.T) - S, ord='fro')**2
# #         return loss

# #     def plot_loss(self):
# #         plt.figure(figsize=(8, 6))
# #         plt.plot(range(self.num_epochs), self.node2vec_loss_history, marker='o', linestyle='-', label='Node2Vec')
# #         plt.plot(range(self.num_epochs), self.word2vec_loss_history, marker='o', linestyle='-', label='Word2Vec')
# #         plt.title("Loss Over Epochs")
# #         plt.xlabel("Epoch")
# #         plt.ylabel("Loss")
# #         plt.legend()
# #         plt.grid(True)
# #         plt.show()

# #     def plot_embeddings_with_pca_tsne(self, node_embeddings):
# #         # Reduce dimensions using PCA
# #         pca = PCA(n_components=2)
# #         pca_result = pca.fit_transform(node_embeddings)

# #         # Further reduce dimensions using t-SNE
# #         tsne = TSNE(n_components=2)
# #         tsne_result = tsne.fit_transform(pca_result)

# #         # Plot t-SNE result
# #         plt.figure(figsize=(10, 8))
# #         plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
# #         plt.title("t-SNE Visualization of Node Embeddings")
# #         plt.show()

# #     def fit(self, graph):
# #         # Step 1: Generate Random Walks
# #         walks = self.generate_random_walks(graph)

# #         # Step 2: Calculate Similarity Matrix
# #         similarity_matrix = self.calculate_similarity_matrix(graph, walks)

# #         # Step 3: Initialize Node Embeddings
# #         nodes = list(graph.nodes)
# #         num_nodes = len(nodes)
# #         self.node_embeddings = np.random.rand(num_nodes, self.dimensions)

# #         # Step 4: Train Node Embeddings
# #         for epoch in tqdm(range(self.num_epochs), desc="Training Node Embeddings"):
# #             for i in range(num_nodes):
# #                 for j in range(i + 1, num_nodes):
# #                     if similarity_matrix[i][j] == 0:
# #                         continue
# #                     diff = self.node_embeddings[i] - self.node_embeddings[j]
# #                     gradient = 2 * similarity_matrix[i][j] * np.outer(diff, diff)
# #                     self.node_embeddings[i] -= gradient
# #                     self.node_embeddings[j] += gradient

# #             # Calculate and store the loss for Node2Vec for this epoch
# #             current_node2vec_loss = self.calculate_loss(self.node_embeddings, similarity_matrix)
# #             self.node2vec_loss_history.append(current_node2vec_loss)

# #         # Step 5: Train Word2Vec
# #         sentences = walks  # Each random walk is treated as a sentence
# #         w2v_model = Word2Vec(sentences, vector_size=self.dimensions, window=5, sg=1, min_count=1, workers=4)
# #         self.word2vec_embeddings = np.array([w2v_model.wv[str(node)] for node in nodes])

# #         # Calculate the Word2Vec loss
# #         w2v_loss = w2v_model.get_latest_training_loss()
# #         self.word2vec_loss_history = [w2v_loss] * self.num_epochs

# #         return self.node_embeddings, self.word2vec_embeddings


# import networkx as nx
# import numpy as np
# import random

# class Node2Vec:
#     def __init__(self, graph, dimensions, walk_length, num_walks, p, q, T, learning_rate, window_size):
#         self.graph = graph
#         self.dimensions = dimensions
#         self.walk_length = walk_length
#         self.num_walks = num_walks
#         self.p = p
#         self.q = q
#         self.T = T  # Length of random walks for similarity
#         self.learning_rate = learning_rate
#         self.window_size = window_size  # Add the window_size attribute

#         # Initialize node embeddings randomly
#         self.node_embeddings = {node: np.random.rand(dimensions) for node in graph.nodes()}


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
#         return np.random.choice(neighbors, p=normalized_weights)

#     def train(self):
#         for _ in range(self.num_walks):
#             for node in self.graph.nodes():
#                 walk = self.node2vec_walk(node)
#                 self.update_embeddings(walk)

#     def update_embeddings(self, walk):
#         for i, node in enumerate(walk):
#             for j in range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)):
#                 if i != j:
#                     node_i = walk[i]
#                     node_j = walk[j]
#                     # Calculate loss and update embeddings based on probability of visit
#                     loss = self.node_similarity_loss(node_i, node_j)
#                     gradient = self.gradient(loss, self.node_embeddings[node_i], self.node_embeddings[node_j])
#                     self.node_embeddings[node_i] -= self.learning_rate * gradient

# #     def node_similarity_loss(self, node_i, node_j):
# #         # Probability of visiting node_j on a length-T random walk starting at node_i
# #         prob_ij = self.get_node_similarity(node_i, node_j)
# #         # Negative log likelihood as the loss
# #         return -np.log(prob_ij)
#     def node_similarity_loss(self, node_i, node_j):
#         # Probability of visiting node_j on a length-T random walk starting at node_i
#         prob_ij = self.get_node_similarity(node_i, node_j)

#         # small epsilon to avoid division by zero
#         epsilon = 1e-8
#         prob_ij = max(prob_ij, epsilon)

#         # Negative log likelihood as the loss
#         return -np.log(prob_ij)

#     def get_node_similarity(self, node_i, node_j):
#         # Calculate the probability of visiting node_j on a length-T random walk starting at node_i
#         # This involves simulating T-length random walks and counting the occurrences of node_j
#         count = 0
#         for _ in range(self.T):
#             walk = self.node2vec_walk(node_i)
#             if node_j in walk:
#                 count += 1
#         return count / self.T

#     def gradient(self, loss, emb_i, emb_j):
#         return (1 / (1 + np.exp(loss))) * (emb_j - emb_i)

#     def get_embeddings(self):
#         return self.node_embeddings


import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm

class Node2Vec(nn.Module):
    def __init__(self, graph, dimensions, walk_length, num_walks, p, q, T, learning_rate, window_size):
        super(Node2Vec, self).__init__()
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.T = T  # Length of random walks for similarity
        self.learning_rate = learning_rate
        self.window_size = window_size

        # Check if GPU is available and set device accordingly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def train(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Use tqdm to create a progress bar with the total number of walks
        pbar = tqdm(total=self.num_walks, desc="Total Walks")
        for _ in range(self.num_walks):
            pbar.update(1)  # Update the progress bar by 1 walk
            for node in self.graph.nodes():
                walk = self.node2vec_walk(node)
                self.update_embeddings(walk, optimizer, criterion)
        pbar.close()  # Close the progress bar when done

    def update_embeddings(self, walk, optimizer, criterion):
        for i, node in enumerate(walk):
            for j in range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)):
                if i != j:
                    node_i = torch.tensor(walk[i]).to(self.device)
                    node_j = torch.tensor(walk[j]).to(self.device)

                    # Calculate loss and update embeddings based on probability of visit
                    loss = self.node_similarity_loss(node_i, node_j)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    def node_similarity_loss(self, node_i, node_j):
        emb_i = self.node_embeddings(node_i)
        emb_j = self.node_embeddings(node_j)

        # Probability of visiting node_j on a length-T random walk starting at node_i
        prob_ij = torch.sigmoid(torch.matmul(emb_i, emb_j.t()))

        # small epsilon to avoid division by zero
        epsilon = 1e-8
        prob_ij = torch.max(prob_ij, torch.tensor(epsilon).to(self.device))

        # Negative log likelihood as the loss
        return -torch.log(prob_ij)

    def get_embeddings(self):
        return {node: self.node_embeddings(torch.tensor(node).to(self.device)).cpu().detach().numpy() for node in self.graph.nodes()}
