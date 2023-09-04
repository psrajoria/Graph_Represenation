import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

# class Node2Vec:
#     def __init__(
#         self,
#         graph,
#         p,
#         q,
#         dimensions,
#         num_walks,
#         walk_length,
#         window_size,
#         num_epochs,
#         workers,
#     ):
#         self.graph = graph
#         self.p = p
#         self.q = q
#         self.dimensions = dimensions
#         self.num_walks = num_walks
#         self.walk_length = walk_length
#         self.window_size = window_size
#         self.num_epochs = num_epochs
#         self.workers = workers

#         self.walks = self.generate_walks()
#         self.model = self.train_embedding()

#     def generate_walks(self):
#         walks = []
#         for _ in range(self.num_walks):
#             for node in self.graph.nodes():
#                 walk = self.node2vec_walk(node)
#                 walks.append(walk)
#         return walks

#     def node2vec_walk(self, start_node):
#         walk = [start_node]
#         while len(walk) < self.walk_length:
#             curr_node = walk[-1]
#             neighbors = list(self.graph.neighbors(curr_node))
#             if len(neighbors) > 0:
#                 if len(walk) == 1:
#                     walk.append(np.random.choice(neighbors))
#                 else:
#                     prev_node = walk[-2]
#                     next_node = self.weighted_choice(neighbors, prev_node, curr_node)
#                     walk.append(next_node)
#             else:
#                 break
#         return walk

#     def weighted_choice(self, neighbors, prev_node, curr_node):
#         weights = []
#         for neighbor in neighbors:
#             if neighbor == prev_node:
#                 weights.append(1 / self.p)
#             elif self.graph.has_edge(curr_node, neighbor):
#                 weights.append(1)
#             else:
#                 weights.append(1 / self.q)
#         weights = np.array(weights)
#         weights /= weights.sum()
#         return np.random.choice(neighbors, p=weights)

#     def embed_all_nodes(self):
#         embeddings = {}
#         for node in self.graph.nodes():
#             embeddings[node] = self.model.wv[node]
#         return embeddings

#     def train_embedding(self):
#         model = Word2Vec(
#             sentences=self.walks,
#             vector_size=self.dimensions,
#             window=self.window_size,
#             sg=1,  # Skip-gram
#             epochs=self.num_epochs,
#             workers=self.workers,
#         )
#         return model


class Node2Vec:
    def __init__(
        self,
        graph,
        dimensions=64,
        num_walks=10,
        walk_length=80,
        window_size=10,
        num_epochs=100,
        workers=4,
        p=1.0,  # Return parameter
        q=1.0,  # In-out parameter
        learning_rate=0.025,
    ):
        self.graph = graph
        self.dimensions = dimensions
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.window_size = window_size
        self.num_epochs = num_epochs
        self.workers = workers
        self.p = p
        self.q = q
        self.learning_rate = learning_rate

        # Generate random walks (training pairs are implicitly created)
        self.walks = self.generate_walks()

        # Train Word2Vec model
        self.model = self.train_embedding()

    def generate_walks(self):
        walks = []
        for _ in range(self.num_walks):
            for node in self.graph.nodes():
                walk = self.node2vec_walk(node)
                walks.append(walk)
        return walks

    def node2vec_walk(self, start_node):
        walk = [start_node]
        while len(walk) < self.walk_length:
            curr_node = walk[-1]
            neighbors = list(self.graph.neighbors(curr_node))
            if len(neighbors) > 0:
                if len(walk) == 1:
                    walk.append(np.random.choice(neighbors))
                else:
                    prev_node = walk[-2]
                    next_node = self.weighted_choice(neighbors, prev_node, curr_node)
                    walk.append(next_node)
            else:
                break
        return walk

    def weighted_choice(self, neighbors, prev_node, curr_node):
        weights = []
        for neighbor in neighbors:
            # Calculate transition probabilities based on p and q
            if neighbor == prev_node:
                weights.append(1.0 / self.p)
            elif self.graph.has_edge(curr_node, neighbor):
                weights.append(1.0)
            else:
                weights.append(1.0 / self.q)
        weights = np.array(weights)
        weights /= weights.sum()

        # Choose the next node based on the computed probabilities
        next_node = np.random.choice(neighbors, p=weights)
        return next_node

    def train_embedding(self):
        # The Word2Vec model from Gensim is used, which internally calculates loss
        model = Word2Vec(
            sentences=self.walks,
            vector_size=self.dimensions,
            window=self.window_size,
            sg=1,  # Skip-gram
            epochs=self.num_epochs,
            workers=self.workers,
            compute_loss=True,  # Loss calculation is handled by Gensim
            alpha=self.learning_rate,  # Learning rate for SGD
        )
        return model

    def plot_loss(self):
        loss_history = self.model.get_latest_training_loss()
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss During Training")
        plt.show()

    def embed_all_nodes(self):
        embeddings = {}
        for node in self.graph.nodes():
            embeddings[node] = self.model.wv[node]
        return embeddings


# import numpy as np
# import networkx as nx
# from gensim.models import Word2Vec

# class Node2Vec:
#     def __init__(
#         self,
#         graph,
#         p,
#         q,
#         dimensions,
#         num_walks,
#         walk_length,
#         window_size,
#         num_epochs,
#         workers,
#     ):
#         self.graph = graph
#         self.p = p
#         self.q = q
#         self.dimensions = dimensions
#         self.num_walks = num_walks
#         self.walk_length = walk_length
#         self.window_size = window_size
#         self.num_epochs = num_epochs
#         self.workers = workers

#     def generate_walks(self):
#         walks = []
#         for _ in range(self.num_walks):
#             for node in self.graph.nodes():
#                 walk = self.node2vec_walk(node)
#                 walks.append(walk)
#         return walks

#     def node2vec_walk(self, start_node):
#         walk = [start_node]
#         while len(walk) < self.walk_length:
#             curr_node = walk[-1]
#             neighbors = list(self.graph.neighbors(curr_node))
#             if len(neighbors) > 0:
#                 if len(walk) == 1:
#                     walk.append(np.random.choice(neighbors))
#                 else:
#                     next_node = self.weighted_choice(neighbors, walk[-2], curr_node)
#                     walk.append(next_node)
#             else:
#                 break
#         return walk

#     def weighted_choice(self, neighbors, prev_node, curr_node):
#         weights = []
#         for neighbor in neighbors:
#             if neighbor == prev_node:
#                 weights.append(1 / self.p)
#             elif self.graph.has_edge(curr_node, neighbor):
#                 weights.append(1)
#             else:
#                 weights.append(1 / self.q)
#         weights = np.array(weights)
#         weights /= weights.sum()
#         return np.random.choice(neighbors, p=weights)

#     def train_embedding(self):
#         walks = self.generate_walks()
#         model = Word2Vec(
#             sentences=walks,
#             vector_size=self.dimensions,
#             window=self.window_size,
#             sg=1,  # Skip-gram
#             epochs=self.num_epochs,
#             workers=self.workers,
#         )
#         return model

#     def embed_all_nodes(self, model=None):
#         if model is None:
#             model = self.train_embedding()
#         embeddings = {node: model.wv[node] for node in self.graph.nodes()}
#         return embeddings




# import numpy as np
# import networkx as nx
# from gensim.models import Word2Vec
# import matplotlib.pyplot as plt

# class Node2VecOptimized:
#     """
#     Node2VecOptimized: A class for generating embeddings of nodes in a graph using Node2Vec.

#     Args:
#         graph (nx.Graph): The input graph for which embeddings will be generated.
#         dimensions (int): The dimensionality of the node embeddings.
#         num_walks (int): The number of random walks to perform for each node.
#         walk_length (int): The length of each random walk.
#         window_size (int): The context window size for Word2Vec.
#         num_epochs (int): The number of training epochs.
#         p (float): Return parameter for biased random walks.
#         q (float): In-out parameter for biased random walks.
#         learning_rate (float): Learning rate for gradient descent.
#         regularization_coeff (float): Coefficient for L2 regularization.
#         workers (int): The number of parallel workers for generating random walks.

#     Attributes:
#         graph (nx.Graph): The input graph.
#         dimensions (int): The dimensionality of the node embeddings.
#         num_walks (int): The number of random walks to perform for each node.
#         walk_length (int): The length of each random walk.
#         window_size (int): The context window size for Word2Vec.
#         num_epochs (int): The number of training epochs.
#         p (float): Return parameter for biased random walks.
#         q (float): In-out parameter for biased random walks.
#         learning_rate (float): Learning rate for gradient descent.
#         regularization_coeff (float): Coefficient for L2 regularization.
#         workers (int): The number of parallel workers for generating random walks.
#         walks (list of list): List of random walks generated for the graph.
#         embeddings (numpy.ndarray): The node embeddings.

#     Methods:
#         generate_walks(): Generate random walks on the graph.
#         weighted_choice(neighbors, prev_node, curr_node): Choose the next node for a random walk.
#         initialize_embeddings(): Initialize node embeddings randomly.
#         train_embedding(): Train the embeddings using the Node2Vec algorithm.
#         compute_similarity_matrix(): Compute the similarity matrix for the graph.
#         plot_loss(loss_history): Plot the loss during training.
#         create_word2vec_embeddings(dimensions, window, sg, epochs): Create Word2Vec embeddings.

#     Example usage:
#     ```python
#     graph = nx.Graph()  # Create your graph here
#     node2vec = Node2VecOptimized(graph)
#     embeddings = node2vec.embed_all_nodes()
#     node2vec.plot_loss(node2vec.loss_history)

#     # Create Word2Vec embeddings
#     word2vec_embeddings = node2vec.create_word2vec_embeddings(dimensions=64, window=10, sg=1, epochs=10)
#     node2vec.plot_loss(node2vec.train_embedding())  # Plot the Node2Vec loss
#     node2vec.plot_loss(node2vec.create_word2vec_embeddings(dimensions=64, window=10, sg=1, epochs=10))  # Plot the Word2Vec loss
#     ```
#     """

#     def __init__(self, graph, dimensions=64, num_walks=10, walk_length=80,
#                  window_size=10, num_epochs=100, p=1.0, q=1.0,
#                  learning_rate=0.025, regularization_coeff=0.1, workers=4):
#         # Initialize class attributes
#         self.graph = graph
#         self.dimensions = dimensions
#         self.num_walks = num_walks
#         self.walk_length = walk_length
#         self.window_size = window_size
#         self.num_epochs = num_epochs
#         self.p = p
#         self.q = q
#         self.learning_rate = learning_rate
#         self.regularization_coeff = regularization_coeff
#         self.workers = workers
#         self.walks = self.generate_walks()  # Generate random walks
#         self.embeddings = self.initialize_embeddings()  # Initialize node embeddings
#         self.loss_history = []  # To store loss during training

#     def generate_walks(self):
#         """
#         Generate random walks on the graph.

#         Returns:
#             list of list: List of random walks generated for the graph.
#         """
#         def node2vec_walk(start_node):
#             walk = [start_node]
#             while len(walk) < self.walk_length:
#                 curr_node = walk[-1]
#                 neighbors = list(self.graph.neighbors(curr_node))
#                 if len(neighbors) > 0:
#                     if len(walk) == 1:
#                         walk.append(np.random.choice(neighbors))
#                     else:
#                         prev_node = walk[-2]
#                         next_node = self.weighted_choice(neighbors, prev_node, curr_node)  # Compute transition probabilities
#                         walk.append(next_node)
#                 else:
#                     break
#             return walk

#         # Generate random walks for all nodes in parallel
#         walks = [node2vec_walk(node) for _ in range(self.num_walks) for node in self.graph.nodes()]
#         return walks

#     def weighted_choice(self, neighbors, prev_node, curr_node):
#         """
#         Choose the next node for a random walk based on weights.

#         Args:
#             neighbors (list): List of neighboring nodes.
#             prev_node (int): Previous node in the random walk.
#             curr_node (int): Current node in the random walk.

#         Returns:
#             int: The next node to visit in the random walk.
#         """
#         weights = [1.0 / self.p if neighbor == prev_node
#                    else 1.0 if self.graph.has_edge(curr_node, neighbor)
#                    else 1.0 / self.q for neighbor in neighbors]  # Transition probabilities
#         weights = np.array(weights)
#         weights /= weights.sum()
#         next_node = np.random.choice(neighbors, p=weights)
#         return next_node

#     def initialize_embeddings(self):
#         """
#         Initialize node embeddings randomly.

#         Returns:
#             numpy.ndarray: Randomly initialized node embeddings.
#         """
#         num_nodes = len(self.graph.nodes())
#         return np.random.rand(num_nodes, self.dimensions)

#     def train_embedding(self):
#         """
#         Train the embeddings using the Node2Vec algorithm.

#         Returns:
#             list: List of loss values during training.
#         """
#         loss_history = []
#         for epoch in range(self.num_epochs):
#             z_transpose = np.transpose(self.embeddings)
#             similarity_matrix = self.compute_similarity_matrix()
#             loss = np.linalg.norm(np.dot(z_transpose, self.embeddings) - similarity_matrix)
#             loss_history.append(loss)

#             gradient = -2 * np.dot(self.embeddings, similarity_matrix) + 2 * np.dot(self.embeddings, np.dot(z_transpose, self.embeddings))
#             gradient += 2 * self.regularization_coeff * self.embeddings

#             self.embeddings -= self.learning_rate * gradient

#             # Early stopping based on loss change
#             if epoch > 0 and abs(loss_history[-1] - loss_history[-2]) < 1e-5:
#                 break

#         self.loss_history = loss_history
#         return loss_history

#     def compute_similarity_matrix(self):
#         """
#         Compute the similarity matrix for the graph based on both structure and Euclidean distance.

#         Returns:
#             numpy.ndarray: The computed similarity matrix.
#         """
#         num_nodes = len(self.graph.nodes())
#         similarity_matrix = np.zeros((num_nodes, num_nodes))

#         # Compute structural similarity (Jaccard similarity)
#         for i in range(num_nodes):
#             for j in range(i, num_nodes):
#                 if i != j:
#                     neighbors_i = set(self.graph.neighbors(i))
#                     neighbors_j = set(self.graph.neighbors(j))
#                     jaccard_similarity = len(neighbors_i.intersection(neighbors_j)) / len(neighbors_i.union(neighbors_j))
#                     similarity_matrix[i, j] = jaccard_similarity
#                     similarity_matrix[j, i] = jaccard_similarity

#         # Compute Euclidean distance-based similarity
#         node_embeddings = self.embeddings
#         for i in range(num_nodes):
#             for j in range(i, num_nodes):
#                 if i != j:
#                     euclidean_distance = np.linalg.norm(node_embeddings[i] - node_embeddings[j])
#                     euclidean_similarity = 1 / (1 + euclidean_distance)  # Inverse of distance as similarity
#                     similarity_matrix[i, j] += euclidean_similarity
#                     similarity_matrix[j, i] += euclidean_similarity

#         return similarity_matrix

#     def plot_loss(self, loss_history):
#         """
#         Plot the loss during training.

#         Args:
#             loss_history (list): List of loss values during training.
#         """
#         plt.figure(figsize=(10, 5))
#         plt.plot(loss_history, label="Loss")
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.legend()
#         plt.title("Loss During Training")
#         plt.show()

#     def create_word2vec_embeddings(self, dimensions=64, window=10, sg=1, epochs=10):
#         """
#         Create Word2Vec embeddings.

#         Args:
#             dimensions (int): The dimensionality of the Word2Vec embeddings.
#             window (int): The context window size for Word2Vec.
#             sg (int): Training algorithm for Word2Vec (skip-gram or CBOW).
#             epochs (int): Number of training epochs for Word2Vec.

#         Returns:
#             numpy.ndarray: Word2Vec embeddings for nodes.
#         """
#         sentences = [list(map(str, walk)) for walk in self.walks]
#         w2v_model = Word2Vec(sentences, vector_size=dimensions, window=window, sg=sg, epochs=epochs)
#         embeddings = np.array([w2v_model.wv[str(node)] for node in range(len(self.graph.nodes()))])
#         return embeddings

# # Example usage:
# graph = nx.Graph()  # Create your graph here
# node2vec = Node2VecOptimized(graph)
# embeddings = node2vec.embed_all_nodes()
# node2vec.plot_loss(node2vec.loss_history)

# # Create Word2Vec embeddings
# word2vec_embeddings = node2vec.create_word2vec_embeddings(dimensions=64, window=10, sg=1, epochs=10)
# node2vec.plot_loss(node2vec.train_embedding())  # Plot the Node2Vec loss
# node2vec.plot_loss(node2vec.create_word2vec_embeddings(dimensions=64, window=10, sg=1, epochs=10))  # Plot the Word2Vec loss
