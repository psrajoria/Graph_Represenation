# import os
# import networkx as nx
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import matplotlib.pyplot as plt
# from concurrent.futures import ProcessPoolExecutor
# from tqdm import tqdm
# from datetime import datetime

# class Node2Vec(nn.Module):
#     """
#     Node2Vec algorithm for learning node embeddings in a graph.

#     Args:
#         graph (nx.Graph): The input graph.
#         dimensions (int): The dimensionality of the node embeddings.
#         walk_length (int): Length of each random walk.
#         num_walks (int): Number of random walks to perform per node.
#         p (float): Return parameter for controlling BFS exploration.
#         q (float): In-out parameter for controlling BFS exploration.
#         T (int): Length of random walks for similarity.
#         learning_rate (float): Learning rate for stochastic gradient descent.
#         window_size (int): Maximum distance between the current and predicted node within a sentence.
#         epochs (int): Number of training epochs.
#         negative_samples (int): Number of negative samples per positive pair.

#     Attributes:
#         graph (nx.Graph): The input graph.
#         dimensions (int): The dimensionality of the node embeddings.
#         walk_length (int): Length of each random walk.
#         num_walks (int): Number of random walks to perform per node.
#         p (float): Return parameter for controlling BFS exploration.
#         q (float): In-out parameter for controlling BFS exploration.
#         T (int): Length of random walks for similarity.
#         learning_rate (float): Learning rate for stochastic gradient descent.
#         window_size (int): Maximum distance between the current and predicted node within a sentence.
#         epochs (int): Number of training epochs.
#         negative_samples (int): Number of negative samples per positive pair.
#         device (torch.device): The device (CPU or GPU) for computation.
#         node_embeddings (nn.Embedding): Node embeddings model.

#     Methods:
#         node2vec_walk(start_node): Generate a single biased random walk starting from the given node.
#         node2vec_step(current_node, previous_node): Compute the next step in a biased random walk.
#         train(): Train the Node2Vec model.
#         update_embeddings(walk, optimizer, criterion): Update node embeddings based on random walks.
#         skipgram_loss(target_node, context_node, optimizer, criterion, negative=False): Calculate the loss for skip-gram with negative sampling.
#         get_embeddings(): Get node embeddings for all nodes in the graph.
#         plot_loss(loss_history): Save and display the training loss plot.
#     """

#     def __init__(self, graph, dimensions, walk_length, num_walks, p, q, T, learning_rate, window_size, epochs, negative_samples):
#         super(Node2Vec, self).__init__()
#         self.graph = graph
#         self.dimensions = dimensions
#         self.walk_length = walk_length
#         self.num_walks = num_walks
#         self.p = p
#         self.q = q
#         self.T = T
#         self.learning_rate = learning_rate
#         self.window_size = window_size
#         self.epochs = epochs
#         self.negative_samples = negative_samples

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.node_embeddings = nn.Embedding(len(graph.nodes()), dimensions).to(self.device)
#         nn.init.xavier_uniform_(self.node_embeddings.weight)

#     def node2vec_walk(self, start_node):
#         """
#         Generate a single biased random walk starting from the given node.

#         Args:
#             start_node: The starting node for the random walk.

#         Returns:
#             list: A list representing the generated random walk.
#         """
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
#         """
#         Compute the next step in a biased random walk.

#         Args:
#             current_node: The current node in the random walk.
#             previous_node: The previous node in the random walk.

#         Returns:
#             int: The next node in the random walk.
#         """
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
#         """
#         Train the Node2Vec model.
#         """
#         optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
#         criterion = nn.CrossEntropyLoss()
        
#         loss_history = []

#         for epoch in range(self.epochs):
#             total_loss = 0.0
#             pbar = tqdm(total=self.num_walks, desc=f"Epoch {epoch + 1}/{self.epochs}")

#             with ProcessPoolExecutor() as executor:
#                 futures = []
#                 for _ in range(self.num_walks):
#                     for node in self.graph.nodes():
#                         futures.append(executor.submit(self.node2vec_walk, node))

#                 for future in tqdm(futures, desc="Generating Random Walks", leave=False):
#                     walk = future.result()
#                     loss = self.update_embeddings(walk, optimizer, criterion)
#                     total_loss += loss.item()

#                 pbar.update(self.num_walks)
#                 pbar.close()
#                 loss_history.append(total_loss / (self.num_walks * len(self.graph.nodes())))

#         self.plot_loss(loss_history)

#     def update_embeddings(self, walk, optimizer, criterion):
#         """
#         Update node embeddings based on random walks using skip-gram with negative sampling.

#         Args:
#             walk (list): The random walk sequence.
#             optimizer: The optimization algorithm.
#             criterion: The loss function.

#         Returns:
#             torch.Tensor: The total loss for the walk.
#         """
#         total_loss = 0.0
#         for i, target_node in enumerate(walk):
#             # Generate positive samples
#             for j in range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)):
#                 if i != j:
#                     context_node = walk[j]
#                     loss = self.skipgram_loss(target_node, context_node, optimizer, criterion)
#                     total_loss += loss.item()

#             # Generate negative samples (negative sampling)
#             for _ in range(self.negative_samples):
#                 negative_context_node = random.choice(list(self.graph.nodes()))
#                 loss = self.skipgram_loss(target_node, negative_context_node, optimizer, criterion, negative=True)
#                 total_loss += loss.item()

#         return total_loss

#     def skipgram_loss(self, target_node, context_node, optimizer, criterion, negative=False):
#         """
#         Calculate the loss for skip-gram with negative sampling.

#         Args:
#             target_node: The embedding of the target node.
#             context_node: The embedding of the context node (positive or negative).
#             optimizer: The optimization algorithm.
#             criterion: The loss function.
#             negative (bool): Whether it's a negative sample (default: False).

#         Returns:
#             torch.Tensor: The loss.
#         """
#         target_embedding = self.node_embeddings(torch.tensor(target_node).to(self.device))
#         context_embedding = self.node_embeddings(torch.tensor(context_node).to(self.device))

#         if negative:
#             # Calculate loss for negative samples (maximize similarity)
#             loss = -torch.log(1 - torch.sigmoid(torch.matmul(target_embedding, context_embedding.t())))
#         else:
#             # Calculate loss for positive samples (minimize similarity)
#             loss = -torch.log(torch.sigmoid(torch.matmul(target_embedding, context_embedding.t())))

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         return loss

#     def get_embeddings(self):
#         """
#         Get node embeddings for all nodes in the graph.

#         Returns:
#             dict: A dictionary mapping nodes to their embeddings.
#         """
#         return {node: self.node_embeddings(torch.tensor(node).to(self.device)).cpu().detach().numpy() for node in self.graph.nodes()}

#     def plot_loss(self, loss_history):
#         """
#         Save and display the training loss plot.

#         Args:
#             loss_history (list): List of loss values for each epoch.
#         """
#         plt.figure(figsize=(10, 5))
#         plt.plot(range(1, self.epochs + 1), loss_history)
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.title('Training Loss')
#         plt.grid(True)

#         # Create a folder to save the plots if it doesn't exist
#         save_folder = "loss_plots"
#         os.makedirs(save_folder, exist_ok=True)

#         # Create a filename based on parameters and date-time
#         timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#         filename = f"{timestamp}_p{self.p}_q{self.q}_walk{self.walk_length}_win{self.window_size}.png"
#         filepath = os.path.join(save_folder, filename)

#         # Save and display the plot
#         plt.savefig(filepath)
#         plt.show()
#         plt.close()



# import os
# import networkx as nx
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# from concurrent.futures import ProcessPoolExecutor
# from tqdm import tqdm
# from datetime import datetime


# class Node2Vec(nn.Module):
#     """
#     Node2Vec algorithm for learning node embeddings in a graph.

#     Args:
#         graph (nx.Graph): The input graph.
#         dimensions (int): The dimensionality of the node embeddings.
#         walk_length (int): Length of each random walk.
#         num_walks (int): Number of random walks to perform per node.
#         p (float): Return parameter for controlling BFS exploration.
#         q (float): In-out parameter for controlling BFS exploration.
#         T (int): Length of random walks for similarity.
#         learning_rate (float): Learning rate for stochastic gradient descent.
#         window_size (int): Maximum distance between the current and predicted node within a sentence.
#         epochs (int): Number of training epochs.
#         negative_samples (int): Number of negative samples per positive pair.

#     Attributes:
#         graph (nx.Graph): The input graph.
#         dimensions (int): The dimensionality of the node embeddings.
#         walk_length (int): Length of each random walk.
#         num_walks (int): Number of random walks to perform per node.
#         p (float): Return parameter for controlling BFS exploration.
#         q (float): In-out parameter for controlling BFS exploration.
#         T (int): Length of random walks for similarity.
#         learning_rate (float): Learning rate for stochastic gradient descent.
#         window_size (int): Maximum distance between the current and predicted node within a sentence.
#         epochs (int): Number of training epochs.
#         negative_samples (int): Number of negative samples per positive pair.
#         device (torch.device): The device (CPU or GPU) for computation.
#         node_embeddings (nn.Embedding): Node embeddings model.

#     Methods:
#         compute_transition_probs(): Compute transition probabilities for each node in the graph.
#         compute_transition_probs_single(current_node, neighbors): Compute transition probabilities for a single node.
#         node2vec_walk(start_node): Generate a single biased random walk starting from the given node.
#         node2vec_step(current_node, previous_node, neighbors, transition_probs):
#             Compute the next step in a biased random walk.
#         train(): Train the Node2Vec model.
#         update_embeddings(walk, optimizer, criterion): Update node embeddings based on random walks.
#         skipgram_loss(target_node, context_node, optimizer, criterion, negative=False):
#             Calculate the loss for skip-gram with negative sampling.
#         get_embeddings(): Get node embeddings for all nodes in the graph.
#         plot_loss(loss_history): Save and display the training loss plot.
#     """

#     def __init__(self, graph, dimensions, walk_length, num_walks, p, q, T, learning_rate, window_size, epochs, negative_samples):
#         super(Node2Vec, self).__init__()
#         self.graph = graph.to_directed()
#         self.dimensions = dimensions
#         self.walk_length = walk_length
#         self.num_walks = num_walks
#         self.p = p
#         self.q = q
#         self.T = T
#         self.learning_rate = learning_rate
#         self.window_size = window_size
#         self.epochs = epochs
#         self.negative_samples = negative_samples

#         # Check if CUDA (GPU) is available, and set the device accordingly
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Initialize node embeddings model
#         self.node_embeddings = nn.Embedding(len(graph.nodes()), dimensions).to(self.device)
#         nn.init.xavier_uniform_(self.node_embeddings.weight)

#         # Precompute transition probabilities for efficient random walks
#         self.transition_probs = self.compute_transition_probs()

#     def compute_transition_probs(self):
#         """
#         Compute transition probabilities for each node in the graph.

#         Returns:
#             dict: A dictionary mapping nodes to their transition probabilities.
#         """
#         transition_probs = {}
#         for node in self.graph.nodes():
#             neighbors = list(self.graph.neighbors(node))
#             probs = self.compute_transition_probs_single(node, neighbors)
#             transition_probs[node] = (neighbors, probs)
#         return transition_probs

#     def compute_transition_probs_single(self, current_node, neighbors):
#         """
#         Compute transition probabilities for a single node.

#         Args:
#             current_node: The current node in the random walk.
#             neighbors: List of neighboring nodes.

#         Returns:
#             list: List of transition probabilities for neighboring nodes.
#         """
#         probs = []
#         for neighbor in neighbors:
#             if neighbor == current_node:
#                 probs.append(1 / self.p)
#             elif neighbor in self.graph[current_node]:
#                 probs.append(1)
#             else:
#                 probs.append(1 / self.q)

#         normalized_probs = np.array(probs) / sum(probs)
#         return normalized_probs

#     def node2vec_walk(self, start_node):
#         """
#         Generate a single biased random walk starting from the given node.

#         Args:
#             start_node: The starting node for the random walk.

#         Returns:
#             list: A list representing the generated random walk.
#         """
#         walk = [start_node]
#         while len(walk) < self.walk_length:
#             current_node = walk[-1]
#             neighbors, transition_probs = self.transition_probs[current_node]
#             if len(neighbors) > 0:
#                 next_node = self.node2vec_step(current_node, walk[-2] if len(walk) > 1 else None, neighbors, transition_probs)
#                 walk.append(next_node)
#             else:
#                 break
#         return walk

#     def node2vec_step(self, current_node, previous_node, neighbors, transition_probs):
#         """
#         Compute the next step in a biased random walk.

#         Args:
#             current_node: The current node in the random walk.
#             previous_node: The previous node in the random walk.
#             neighbors: List of neighboring nodes.
#             transition_probs: List of transition probabilities for neighboring nodes.

#         Returns:
#             int: The next node in the random walk.
#         """
#         if previous_node is None:
#             return np.random.choice(neighbors)

#         weights = transition_probs
#         return np.random.choice(neighbors, p=weights)

#     def train(self):
#         """
#         Train the Node2Vec model.

#         This method performs the training process, which includes generating random walks, updating node embeddings, and tracking loss.

#         Returns:
#             None
#         """
#         optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
#         criterion = nn.CrossEntropyLoss()
#         loss_history = []

#         for epoch in range(self.epochs):
#             total_loss = 0.0
#             pbar = tqdm(total=self.num_walks * len(self.graph.nodes()), desc=f"Epoch {epoch + 1}/{self.epochs}")

#             with ProcessPoolExecutor() as executor:
#                 futures = []
#                 for _ in range(self.num_walks):
#                     for node in self.graph.nodes():
#                         futures.append(executor.submit(self.node2vec_walk, node))

#                 for future in tqdm(futures, desc="Generating Random Walks", leave=False):
#                     walk = future.result()
#                     loss = self.update_embeddings(walk, optimizer, criterion)
#                     total_loss += loss.item()
#                     pbar.update(1)

#                 pbar.close()
#                 loss_history.append(total_loss / (self.num_walks * len(self.graph.nodes())))

#         self.plot_loss(loss_history)

#     def update_embeddings(self, walk, optimizer, criterion):
#         """
#         Update node embeddings based on random walks using skip-gram with negative sampling.

#         Args:
#             walk (list): The random walk sequence.
#             optimizer: The optimization algorithm.
#             criterion: The loss function.

#         Returns:
#             torch.Tensor: The total loss for the walk.
#         """
#         total_loss = 0.0
#         pbar = tqdm(total=len(walk), desc="Updating Embeddings", leave=False)

#         for i, target_node in enumerate(walk):
#             positive_samples = [walk[j] for j in range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)) if i != j]
#             for context_node in positive_samples:
#                 loss = self.skipgram_loss(target_node, context_node, optimizer, criterion)
#                 total_loss += loss.item()

#             negative_samples = [np.random.choice(list(self.graph.nodes())) for _ in range(self.negative_samples)]
#             for negative_context_node in negative_samples:
#                 loss = self.skipgram_loss(target_node, negative_context_node, optimizer, criterion, negative=True)
#                 total_loss += loss.item()
#             pbar.update(1)

#         pbar.close()
#         return total_loss

# #     def skipgram_loss(self, target_node, context_node, optimizer, criterion, negative=False):
# #         """
# #         Calculate the loss for skip-gram with negative sampling.

# #         Args:
# #             target_node: The embedding of the target node.
# #             context_node: The embedding of the context node (positive or negative).
# #             optimizer: The optimization algorithm.
# #             criterion: The loss function.
# #             negative (bool): Whether it's a negative sample (default: False).

# #         Returns:
# #             torch.Tensor: The loss.
# #         """
# #         # No need to call .to(self.device) here since the device was set during initialization
# #         target_embedding = self.node_embeddings(torch.tensor(target_node))
# #         context_embedding = self.node_embeddings(torch.tensor(context_node))

# #         if negative:
# #             loss = -torch.log(1 - torch.sigmoid(torch.matmul(target_embedding, context_embedding.t())))
# #         else:
# #             loss = -torch.log(torch.sigmoid(torch.matmul(target_embedding, context_embedding.t())))

# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()

# #         return loss

#     def skipgram_loss(self, target_node, context_node, optimizer, criterion, negative=False):
#         """
#         Calculate the loss for skip-gram with negative sampling.

#         Args:
#             target_node: The embedding of the target node.
#             context_node: The embedding of the context node (positive or negative).
#             optimizer: The optimization algorithm.
#             criterion: The loss function.
#             negative (bool): Whether it's a negative sample (default: False).

#         Returns:
#             torch.Tensor: The loss.
#         """
#         # Move both target_embedding and context_embedding to the same device
#         target_embedding = self.node_embeddings(torch.tensor(target_node).to(self.device))
#         context_embedding = self.node_embeddings(torch.tensor(context_node).to(self.device))

#         if negative:
#             loss = -torch.log(1 - torch.sigmoid(torch.matmul(target_embedding, context_embedding.t())))
#         else:
#             loss = -torch.log(torch.sigmoid(torch.matmul(target_embedding, context_embedding.t())))

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         return loss

#     def get_embeddings(self):
#         """
#         Get node embeddings for all nodes in the graph.

#         Returns:
#             dict: A dictionary mapping nodes to their embeddings.
#         """
#         return {node: self.node_embeddings(torch.tensor(node)).cpu().detach().numpy() for node in self.graph.nodes()}

#     def plot_loss(self, loss_history):
#         """
#         Save and display the training loss plot.

#         Args:
#             loss_history (list): List of loss values for each epoch.
#         """
#         plt.figure(figsize=(10, 5))
#         plt.plot(range(1, self.epochs + 1), loss_history)
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.title('Training Loss')
#         plt.grid(True)

#         save_folder = "loss_plots"
#         os.makedirs(save_folder, exist_ok=True)

#         timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#         filename = f"{timestamp}_p{self.p}_q{self.q}_walk{self.walk_length}_win{self.window_size}.png"
#         filepath = os.path.join(save_folder, filename)

#         plt.savefig(filepath)
#         plt.show()
#         plt.close()


# import os
# import networkx as nx
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# from concurrent.futures import ProcessPoolExecutor
# from tqdm import tqdm
# from datetime import datetime

# class Node2Vec(nn.Module):
#     """
#     Node2Vec algorithm for learning node embeddings in a graph.

#     Args:
#         graph (nx.Graph): The input graph.
#         dimensions (int): The dimensionality of the node embeddings.
#         walk_length (int): Length of each random walk.
#         num_walks (int): Number of random walks to perform per node.
#         p (float): Return parameter for controlling BFS exploration.
#         q (float): In-out parameter for controlling BFS exploration.
#         T (int): Length of random walks for similarity.
#         learning_rate (float): Learning rate for stochastic gradient descent.
#         window_size (int): Maximum distance between the current and predicted node within a sentence.
#         epochs (int): Number of training epochs.
#         negative_samples (int): Number of negative samples per positive pair.

#     Attributes:
#         graph (nx.Graph): The input graph.
#         dimensions (int): The dimensionality of the node embeddings.
#         walk_length (int): Length of each random walk.
#         num_walks (int): Number of random walks to perform per node.
#         p (float): Return parameter for controlling BFS exploration.
#         q (float): In-out parameter for controlling BFS exploration.
#         T (int): Length of random walks for similarity.
#         learning_rate (float): Learning rate for stochastic gradient descent.
#         window_size (int): Maximum distance between the current and predicted node within a sentence.
#         epochs (int): Number of training epochs.
#         negative_samples (int): Number of negative samples per positive pair.
#         device (torch.device): The device (CPU or GPU) for computation.
#         node_embeddings (nn.Embedding): Node embeddings model.

#     Methods:
#         compute_transition_probs(): Compute transition probabilities for each node in the graph.
#         compute_transition_probs_single(current_node, neighbors): Compute transition probabilities for a single node.
#         node2vec_walk(start_node): Generate a single biased random walk starting from the given node.
#         node2vec_step(current_node, previous_node, neighbors, transition_probs):
#             Compute the next step in a biased random walk.
#         train(): Train the Node2Vec model.
#         update_embeddings(walk, optimizer, criterion): Update node embeddings based on random walks.
#         skipgram_loss(target_node, context_node, optimizer, criterion, negative=False):
#             Calculate the loss for skip-gram with negative sampling.
#         get_embeddings(): Get node embeddings for all nodes in the graph.
#         plot_loss(loss_history): Save and display the training loss plot.
#     """

#     def __init__(self, graph, dimensions, walk_length, num_walks, p, q, T, learning_rate, window_size, epochs, negative_samples):
#         super(Node2Vec, self).__init__()
#         self.graph = graph.to_directed()
#         self.dimensions = dimensions
#         self.walk_length = walk_length
#         self.num_walks = num_walks
#         self.p = p
#         self.q = q
#         self.T = T
#         self.learning_rate = learning_rate
#         self.window_size = window_size
#         self.epochs = epochs
#         self.negative_samples = negative_samples

#         # Check if CUDA (GPU) is available, and set the device accordingly
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Initialize node embeddings model
#         self.node_embeddings = nn.Embedding(len(graph.nodes()), dimensions).to(self.device)
#         nn.init.xavier_uniform_(self.node_embeddings.weight)

#         # Precompute transition probabilities for efficient random walks
#         self.transition_probs = self.compute_transition_probs()

#     def compute_transition_probs(self):
#         """
#         Compute transition probabilities for each node in the graph.

#         Returns:
#             dict: A dictionary mapping nodes to their transition probabilities.
#         """
#         transition_probs = {}
#         for node in self.graph.nodes():
#             neighbors = list(self.graph.neighbors(node))
#             probs = self.compute_transition_probs_single(node, neighbors)
#             transition_probs[node] = (neighbors, probs)
#         return transition_probs

#     def compute_transition_probs_single(self, current_node, neighbors):
#         """
#         Compute transition probabilities for a single node.

#         Args:
#             current_node: The current node in the random walk.
#             neighbors: List of neighboring nodes.

#         Returns:
#             list: List of transition probabilities for neighboring nodes.
#         """
#         probs = []
#         for neighbor in neighbors:
#             if neighbor == current_node:
#                 probs.append(1 / self.p)
#             elif neighbor in self.graph[current_node]:
#                 probs.append(1)
#             else:
#                 probs.append(1 / self.q)

#         normalized_probs = torch.Tensor(probs) / torch.sum(torch.Tensor(probs))
#         return normalized_probs

#     def node2vec_walk(self, start_node):
#         """
#         Generate a single biased random walk starting from the given node.

#         Args:
#             start_node: The starting node for the random walk.

#         Returns:
#             list: A list representing the generated random walk.
#         """
#         walk = [start_node]
#         while len(walk) < self.walk_length:
#             current_node = walk[-1]
#             neighbors, transition_probs = self.transition_probs[current_node]
#             if len(neighbors) > 0:
#                 next_node = self.node2vec_step(current_node, walk[-2] if len(walk) > 1 else None, neighbors, transition_probs)
#                 walk.append(next_node)
#             else:
#                 break
#         return walk

#     def node2vec_step(self, current_node, previous_node, neighbors, transition_probs):
#         """
#         Compute the next step in a biased random walk.

#         Args:
#             current_node: The current node in the random walk.
#             previous_node: The previous node in the random walk.
#             neighbors: List of neighboring nodes.
#             transition_probs: List of transition probabilities for neighboring nodes.

#         Returns:
#             int: The next node in the random walk.
#         """
#         if previous_node is None:
#             return torch.choice(neighbors)

#         weights = transition_probs
#         return torch.choice(neighbors, p=weights)

#     def train(self):
#         """
#         Train the Node2Vec model.

#         This method performs the training process, which includes generating random walks, updating node embeddings, and tracking loss.

#         Returns:
#             None
#         """
#         optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
#         criterion = nn.CrossEntropyLoss()
#         loss_history = []

#         for epoch in range(self.epochs):
#             total_loss = 0.0
#             pbar = tqdm(total=self.num_walks * len(self.graph.nodes()), desc=f"Epoch {epoch + 1}/{self.epochs}")

#             with ProcessPoolExecutor() as executor:
#                 futures = []
#                 for _ in range(self.num_walks):
#                     for node in self.graph.nodes():
#                         futures.append(executor.submit(self.node2vec_walk, node))

#                 for future in tqdm(futures, desc="Generating Random Walks", leave=False):
#                     walk = future.result()
#                     loss = self.update_embeddings(walk, optimizer, criterion)
#                     total_loss += loss.item()
#                     pbar.update(1)

#                 pbar.close()
#                 loss_history.append(total_loss / (self.num_walks * len(self.graph.nodes())))

#         self.plot_loss(loss_history)

#     def update_embeddings(self, walk, optimizer, criterion):
#         """
#         Update node embeddings based on random walks using skip-gram with negative sampling.

#         Args:
#             walk (list): The random walk sequence.
#             optimizer: The optimization algorithm.
#             criterion: The loss function.

#         Returns:
#             torch.Tensor: The total loss for the walk.
#         """
#         total_loss = 0.0
#         pbar = tqdm(total=len(walk), desc="Updating Embeddings", leave=False)

#         for i, target_node in enumerate(walk):
#             positive_samples = [walk[j] for j in range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)) if i != j]
#             for context_node in positive_samples:
#                 loss = self.skipgram_loss(target_node, context_node, optimizer, criterion)
#                 total_loss += loss.item()

#             negative_samples = [torch.randint(len(self.graph.nodes()), (1,), dtype=torch.long).item() for _ in range(self.negative_samples)]
#             for negative_context_node in negative_samples:
#                 loss = self.skipgram_loss(target_node, negative_context_node, optimizer, criterion, negative=True)
#                 total_loss += loss.item()
#             pbar.update(1)

#         pbar.close()
#         return total_loss

#     def skipgram_loss(self, target_node, context_node, optimizer, criterion, negative=False):
#         """
#         Calculate the loss for skip-gram with negative sampling.

#         Args:
#             target_node: The embedding of the target node.
#             context_node: The embedding of the context node (positive or negative).
#             optimizer: The optimization algorithm.
#             criterion: The loss function.
#             negative (bool): Whether it's a negative sample (default: False).

#         Returns:
#             torch.Tensor: The loss.
#         """
#         # Move both target_embedding and context_embedding to the same device
#         target_embedding = self.node_embeddings(torch.tensor(target_node).to(self.device))
#         context_embedding = self.node_embeddings(torch.tensor(context_node).to(self.device))

#         if negative:
#             loss = -torch.log(1 - torch.sigmoid(torch.matmul(target_embedding, context_embedding.t())))
#         else:
#             loss = -torch.log(torch.sigmoid(torch.matmul(target_embedding, context_embedding.t())))

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         return loss

#     def get_embeddings(self):
#         """
#         Get node embeddings for all nodes in the graph.

#         Returns:
#             dict: A dictionary mapping nodes to their embeddings.
#         """
#         return {node: self.node_embeddings(torch.tensor(node)).cpu().detach().numpy() for node in self.graph.nodes()}

#     def plot_loss(self, loss_history):
#         """
#         Save and display the training loss plot.

#         Args:
#             loss_history (list): List of loss values for each epoch.
#         """
#         plt.figure(figsize=(10, 5))
#         plt.plot(range(1, self.epochs + 1), loss_history)
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.title('Training Loss')
#         plt.grid(True)

#         save_folder = "loss_plots"
#         os.makedirs(save_folder, exist_ok=True)

#         timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#         filename = f"{timestamp}_p{self.p}_q{self.q}_walk{self.walk_length}_win{self.window_size}.png"
#         filepath = os.path.join(save_folder, filename)

#         plt.savefig(filepath)
#         plt.show()
#         plt.close()

# # Example usage:
# # Define your graph and parameters
# # graph = nx.Graph()  # Replace with your graph
# # dimensions = 128
# # walk_length = 80
# # num_walks = 10
# # p = 1.0
# # q = 1.0
# # T = 10
# # learning_rate = 0.025
# # window_size = 10
# # epochs = 3
# # negative_samples = 5

# # node2vec = Node2Vec(graph, dimensions, walk_length, num_walks, p, q, T, learning_rate, window_size, epochs, negative_samples)
# # node2vec.train()
# # embeddings = node2vec.get_embeddings()


# import os
# import networkx as nx
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# from concurrent.futures import ProcessPoolExecutor
# from tqdm import tqdm
# from datetime import datetime


# class Node2Vec(nn.Module):
#     """
#     Node2Vec algorithm for learning node embeddings in a graph.

#     Args:
#         graph (nx.Graph): The input graph.
#         dimensions (int): The dimensionality of the node embeddings.
#         walk_length (int): Length of each random walk.
#         num_walks (int): Number of random walks to perform per node.
#         p (float): Return parameter for controlling BFS exploration.
#         q (float): In-out parameter for controlling BFS exploration.
#         T (int): Length of random walks for similarity.
#         learning_rate (float): Learning rate for stochastic gradient descent.
#         window_size (int): Maximum distance between the current and predicted node within a sentence.
#         epochs (int): Number of training epochs.
#         negative_samples (int): Number of negative samples per positive pair.

#     Attributes:
#         graph (nx.Graph): The input graph.
#         dimensions (int): The dimensionality of the node embeddings.
#         walk_length (int): Length of each random walk.
#         num_walks (int): Number of random walks to perform per node.
#         p (float): Return parameter for controlling BFS exploration.
#         q (float): In-out parameter for controlling BFS exploration.
#         T (int): Length of random walks for similarity.
#         learning_rate (float): Learning rate for stochastic gradient descent.
#         window_size (int): Maximum distance between the current and predicted node within a sentence.
#         epochs (int): Number of training epochs.
#         negative_samples (int): Number of negative samples per positive pair.
#         device (torch.device): The device (CPU or GPU) for computation.
#         node_embeddings (nn.Embedding): Node embeddings model.

#     Methods:
#         compute_transition_probs(): Compute transition probabilities for each node in the graph.
#         compute_transition_probs_single(current_node, neighbors): Compute transition probabilities for a single node.
#         node2vec_walk(start_node): Generate a single biased random walk starting from the given node.
#         node2vec_step(current_node, previous_node, neighbors, transition_probs):
#             Compute the next step in a biased random walk.
#         train(): Train the Node2Vec model.
#         update_embeddings(walk, optimizer, criterion): Update node embeddings based on random walks.
#         skipgram_loss(target_node, context_node, optimizer, criterion, negative=False):
#             Calculate the loss for skip-gram with negative sampling.
#         get_embeddings(): Get node embeddings for all nodes in the graph.
#         plot_loss(loss_history): Save and display the training loss plot.
#     """

#     def __init__(self, graph, dimensions, walk_length, num_walks, p, q, T, learning_rate, window_size, epochs,
#                  negative_samples):
#         super(Node2Vec, self).__init__()
#         self.graph = graph.to_directed()
#         self.dimensions = dimensions
#         self.walk_length = walk_length
#         self.num_walks = num_walks
#         self.p = p
#         self.q = q
#         self.T = T
#         self.learning_rate = learning_rate
#         self.window_size = window_size
#         self.epochs = epochs
#         self.negative_samples = negative_samples

#         # Check if CUDA (GPU) is available, and set the device accordingly
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Initialize node embeddings model
#         self.node_embeddings = nn.Embedding(len(graph.nodes()), dimensions).to(self.device)
#         nn.init.xavier_uniform_(self.node_embeddings.weight)

#         # Precompute transition probabilities for efficient random walks
#         self.transition_probs = self.compute_transition_probs()

#     def compute_transition_probs(self):
#         """
#         Compute transition probabilities for each node in the graph.

#         Returns:
#             dict: A dictionary mapping nodes to their transition probabilities.
#         """
#         transition_probs = {}
#         for node in self.graph.nodes():
#             neighbors = list(self.graph.neighbors(node))
#             probs = self.compute_transition_probs_single(node, neighbors)
#             transition_probs[node] = (neighbors, probs)
#         return transition_probs

#     def compute_transition_probs_single(self, current_node, neighbors):
#         """
#         Compute transition probabilities for a single node.

#         Args:
#             current_node: The current node in the random walk.
#             neighbors: List of neighboring nodes.

#         Returns:
#             list: List of transition probabilities for neighboring nodes.
#         """
#         probs = []
#         for neighbor in neighbors:
#             if neighbor == current_node:
#                 probs.append(1 / self.p)
#             elif neighbor in self.graph[current_node]:
#                 probs.append(1)
#             else:
#                 probs.append(1 / self.q)

#         normalized_probs = torch.Tensor(probs) / torch.sum(torch.Tensor(probs))
#         return normalized_probs

#     def node2vec_walk(self, start_node):
#         """
#         Generate a single biased random walk starting from the given node.

#         Args:
#             start_node: The starting node for the random walk.

#         Returns:
#             list: A list representing the generated random walk.
#         """
#         walk = [start_node]
#         while len(walk) < self.walk_length:
#             current_node = walk[-1]
#             neighbors, transition_probs = self.transition_probs[current_node]
#             if len(neighbors) > 0:
#                 next_node = self.node2vec_step(current_node, walk[-2] if len(walk) > 1 else None, neighbors,
#                                                transition_probs)
#                 walk.append(next_node)
#             else:
#                 break
#         return walk

# #     def node2vec_step(self, current_node, previous_node, neighbors, transition_probs):
# #         """
# #         Compute the next step in a biased random walk.

# #         Args:
# #             current_node: The current node in the random walk.
# #             previous_node: The previous node in the random walk.
# #             neighbors: List of neighboring nodes.
# #             transition_probs: List of transition probabilities for neighboring nodes.

# #         Returns:
# #             int: The next node in the random walk.
# #         """
# #         if previous_node is None:
# #             return torch.choice(neighbors)

# #         # Use torch.randint to randomly select an index
# #         rand_index = torch.randint(len(neighbors), (1,)).item()
# #         return neighbors[rand_index]

#     def node2vec_step(self, current_node, previous_node, neighbors, transition_probs):
#         """
#         Compute the next step in a biased random walk.

#         Args:
#             current_node: The current node in the random walk.
#             previous_node: The previous node in the random walk.
#             neighbors: List of neighboring nodes.
#             transition_probs: List of transition probabilities for neighboring nodes.

#         Returns:
#             int: The next node in the random walk.
#         """
#         if previous_node is None:
#             rand_index = torch.randint(len(neighbors), (1,), dtype=torch.long).item()
#             return neighbors[rand_index]

#         rand_index = torch.randint(len(neighbors), (1,), dtype=torch.long).item()
#         return neighbors[rand_index]

#     def train(self):
#         """
#         Train the Node2Vec model.

#         This method performs the training process, which includes generating random walks, updating node embeddings,
#         and tracking loss.

#         Returns:
#             None
#         """
#         optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
#         criterion = nn.CrossEntropyLoss()
#         loss_history = []

#         for epoch in range(self.epochs):
#             total_loss = 0.0
#             pbar = tqdm(total=self.num_walks * len(self.graph.nodes()), desc=f"Epoch {epoch + 1}/{self.epochs}")

#             with ProcessPoolExecutor() as executor:
#                 futures = []
#                 for _ in range(self.num_walks):
#                     for node in self.graph.nodes():
#                         futures.append(executor.submit(self.node2vec_walk, node))

#                 for future in tqdm(futures, desc="Generating Random Walks", leave=False):
#                     walk = future.result()
#                     loss = self.update_embeddings(walk, optimizer, criterion)
#                     total_loss += loss.item()
#                     pbar.update(1)

#                 pbar.close()
#                 loss_history.append(total_loss / (self.num_walks * len(self.graph.nodes())))

#         self.plot_loss(loss_history)

#     def update_embeddings(self, walk, optimizer, criterion):
#         """
#         Update node embeddings based on random walks using skip-gram with negative sampling.

#         Args:
#             walk (list): The random walk sequence.
#             optimizer: The optimization algorithm.
#             criterion: The loss function.

#         Returns:
#             torch.Tensor: The total loss for the walk.
#         """
#         total_loss = 0.0
#         pbar = tqdm(total=len(walk), desc="Updating Embeddings", leave=False)

#         for i, target_node in enumerate(walk):
#             positive_samples = [walk[j] for j in
#                                 range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)) if
#                                 i != j]
#             for context_node in positive_samples:
#                 loss = self.skipgram_loss(target_node, context_node, optimizer, criterion)
#                 total_loss += loss.item()

#             negative_samples = [torch.randint(len(self.graph.nodes()), (1,), dtype=torch.long).item() for _ in
#                                 range(self.negative_samples)]
#             for negative_context_node in negative_samples:
#                 loss = self.skipgram_loss(target_node, negative_context_node, optimizer, criterion, negative=True)
#                 total_loss += loss.item()
#             pbar.update(1)

#         pbar.close()
#         return total_loss

# #     def skipgram_loss(self, target_node, context_node, optimizer, criterion, negative=False):
# #         """
# #         Calculate the loss for skip-gram with negative sampling.

# #         Args:
# #             target_node: The embedding of the target node.
# #             context_node: The embedding of the context node (positive or negative).
# #             optimizer: The optimization algorithm.
# #             criterion: The loss function.
# #             negative (bool): Whether it's a negative sample (default: False).

# #         Returns:
# #             torch.Tensor: The loss.
# #         """
# #         # Move both target_embedding and context_embedding to the same device
# #         target_embedding = self.node_embeddings(torch.tensor(target_node).to(self.device))
# #         context_embedding = self.node_embeddings(torch.tensor(context_node).to(self.device))

# #         if negative:
# #             loss = -torch.log(1 - torch.sigmoid(torch.matmul(target_embedding, context_embedding.t())))
# #         else:
# #             loss = -torch.log(torch.sigmoid(torch.matmul(target_embedding, context_embedding.t())))

# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()

# #         return loss

#     def skipgram_loss(self, target_node, context_node, optimizer, criterion, negative=False):
#         """
#         Calculate the loss for skip-gram with negative sampling.

#         Args:
#             target_node: The embedding of the target node.
#             context_node: The embedding of the context node (positive or negative).
#             optimizer: The optimization algorithm.
#             criterion: The loss function.
#             negative (bool): Whether it's a negative sample (default: False).

#         Returns:
#             torch.Tensor: The loss.
#         """
#         # Move both target_embedding and context_embedding to the same device
#         target_embedding = self.node_embeddings(torch.tensor(target_node, dtype=torch.long).to(self.device))
#         context_embedding = self.node_embeddings(torch.tensor(context_node, dtype=torch.long).to(self.device))

#         if negative:
#             loss = -torch.log(1 - torch.sigmoid(torch.sum(target_embedding * context_embedding)))
#         else:
#             loss = -torch.log(torch.sigmoid(torch.sum(target_embedding * context_embedding)))

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         return loss

#     def get_embeddings(self):
#         """
#         Get node embeddings for all nodes in the graph.

#         Returns:
#             dict: A dictionary mapping nodes to their embeddings.
#         """
#         return {node: self.node_embeddings(torch.tensor(node)).cpu().detach().numpy() for node in self.graph.nodes()}

#     def plot_loss(self, loss_history):
#         """
#         Save and display the training loss plot.

#         Args:
#             loss_history (list): List of loss values for each epoch.
#         """
#         plt.figure(figsize=(10, 5))
#         plt.plot(range(1, self.epochs + 1), loss_history)
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.title('Training Loss')
#         plt.grid(True)

#         save_folder = "loss_plots"
#         os.makedirs(save_folder, exist_ok=True)

#         timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#         filename = f"{timestamp}_p{self.p}_q{self.q}_walk{self.walk_length}_win{self.window_size}.png"
#         filepath = os.path.join(save_folder, filename)

#         plt.savefig(filepath)
#         plt.show()
#         plt.close()


import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from datetime import datetime

class Node2Vec:
    """
    Node2Vec algorithm for learning node embeddings in a graph.
    (Modified to use CPU and NumPy)

    Args:
        graph (nx.Graph): The input graph.
        dimensions (int): The dimensionality of the node embeddings.
        walk_length (int): Length of each random walk.
        num_walks (int): Number of random walks to perform per node.
        p (float): Return parameter for controlling BFS exploration.
        q (float): In-out parameter for controlling BFS exploration.
        T (int): Length of random walks for similarity.
        learning_rate (float): Learning rate for stochastic gradient descent.
        window_size (int): Maximum distance between the current and predicted node within a sentence.
        epochs (int): Number of training epochs.
        negative_samples (int): Number of negative samples per positive pair.

    Attributes:
        graph (nx.Graph): The input graph.
        dimensions (int): The dimensionality of the node embeddings.
        walk_length (int): Length of each random walk.
        num_walks (int): Number of random walks to perform per node.
        p (float): Return parameter for controlling BFS exploration.
        q (float): In-out parameter for controlling BFS exploration.
        T (int): Length of random walks for similarity.
        learning_rate (float): Learning rate for stochastic gradient descent.
        window_size (int): Maximum distance between the current and predicted node within a sentence.
        epochs (int): Number of training epochs.
        negative_samples (int): Number of negative samples per positive pair.
        device (str): The device ('cpu') for computation.
        node_embeddings (numpy.ndarray): Node embeddings model.
        transition_probs (dict): Precomputed transition probabilities for random walks.

    Methods:
        compute_transition_probs(): Compute transition probabilities for each node in the graph.
        compute_transition_probs_single(current_node, neighbors): Compute transition probabilities for a single node.
        node2vec_walk(start_node): Generate a single biased random walk starting from the given node.
        node2vec_step(current_node, previous_node, neighbors, transition_probs):
            Compute the next step in a biased random walk.
        train(): Train the Node2Vec model.
        update_embeddings(walk): Update node embeddings based on random walks.
        skipgram_loss(target_node, context_node, negative=False):
            Calculate the loss for skip-gram with negative sampling.
        get_embeddings(): Get node embeddings for all nodes in the graph.
        plot_loss(loss_history): Save and display the training loss plot.
    """

    def __init__(self, graph, dimensions, walk_length, num_walks, p, q, T, learning_rate, window_size, epochs,
                 negative_samples):
        self.graph = graph.to_directed()
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.T = T
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.epochs = epochs
        self.negative_samples = negative_samples

        # Set the device to CPU
        self.device = 'cpu'

        # Initialize node embeddings (using NumPy)
        self.node_embeddings = np.random.rand(len(graph.nodes()), dimensions)

        # Precompute transition probabilities for efficient random walks
        self.transition_probs = self.compute_transition_probs()

    def compute_transition_probs(self):
        """
        Compute transition probabilities for each node in the graph.

        Returns:
            dict: A dictionary mapping nodes to their transition probabilities.
        """
        transition_probs = {}
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            probs = self.compute_transition_probs_single(node, neighbors)
            transition_probs[node] = (neighbors, probs)
        return transition_probs

    def compute_transition_probs_single(self, current_node, neighbors):
        """
        Compute transition probabilities for a single node.

        Args:
            current_node: The current node in the random walk.
            neighbors: List of neighboring nodes.

        Returns:
            list: List of transition probabilities for neighboring nodes.
        """
        probs = []
        for neighbor in neighbors:
            if neighbor == current_node:
                probs.append(1 / self.p)
            elif neighbor in self.graph[current_node]:
                probs.append(1)
            else:
                probs.append(1 / self.q)

        normalized_probs = np.array(probs) / np.sum(probs)
        return normalized_probs

    def node2vec_walk(self, start_node):
        """
        Generate a single biased random walk starting from the given node.

        Args:
            start_node: The starting node for the random walk.

        Returns:
            list: A list representing the generated random walk.
        """
        walk = [start_node]
        while len(walk) < self.walk_length:
            current_node = walk[-1]
            neighbors, transition_probs = self.transition_probs[current_node]
            if len(neighbors) > 0:
                next_node = self.node2vec_step(current_node, walk[-2] if len(walk) > 1 else None, neighbors,
                                               transition_probs)
                walk.append(next_node)
            else:
                break
        return walk

    def node2vec_step(self, current_node, previous_node, neighbors, transition_probs):
        """
        Compute the next step in a biased random walk.

        Args:
            current_node: The current node in the random walk.
            previous_node: The previous node in the random walk.
            neighbors: List of neighboring nodes.
            transition_probs: List of transition probabilities for neighboring nodes.

        Returns:
            int: The next node in the random walk.
        """
        if previous_node is None:
            rand_index = np.random.choice(len(neighbors))
            return neighbors[rand_index]

        rand_index = np.random.choice(len(neighbors))
        return neighbors[rand_index]

    def train(self):
        """
        Train the Node2Vec model.

        This method performs the training process, which includes generating random walks, updating node embeddings,
        and tracking loss.

        Returns:
            None
        """
        loss_history = []

        for epoch in range(self.epochs):
            total_loss = 0.0
            pbar = tqdm(total=self.num_walks * len(self.graph.nodes()), desc=f"Epoch {epoch + 1}/{self.epochs}")

            with ProcessPoolExecutor() as executor:
                futures = []
                for _ in range(self.num_walks):
                    for node in self.graph.nodes():
                        futures.append(executor.submit(self.node2vec_walk, node))

                for future in tqdm(futures, desc="Generating Random Walks", leave=False):
                    walk = future.result()
                    loss = self.update_embeddings(walk)
                    total_loss += loss
                    pbar.update(1)

                pbar.close()
                loss_history.append(total_loss / (self.num_walks * len(self.graph.nodes())))

        self.plot_loss(loss_history)

    def update_embeddings(self, walk):
        """
        Update node embeddings based on random walks.

        Args:
            walk (list): The random walk sequence.

        Returns:
            float: The total loss for the walk.
        """
        total_loss = 0.0
        pbar = tqdm(total=len(walk), desc="Updating Embeddings", leave=False)

        for i, target_node in enumerate(walk):
            positive_samples = [walk[j] for j in
                                range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)) if
                                i != j]
            for context_node in positive_samples:
                loss = self.skipgram_loss(target_node, context_node)
                total_loss += loss

            negative_samples = np.random.choice(len(self.graph.nodes()), size=self.negative_samples)
            for negative_context_node in negative_samples:
                loss = self.skipgram_loss(target_node, negative_context_node, negative=True)
                total_loss += loss
            pbar.update(1)

        pbar.close()
        return total_loss

#     def skipgram_loss(self, target_node, context_node, negative=False):
#         """
#         Calculate the loss for skip-gram with negative sampling.

#         Args:
#             target_node: The embedding of the target node.
#             context_node: The embedding of the context node (positive or negative).
#             negative (bool): Whether it's a negative sample (default: False).

#         Returns:
#             float: The loss.
#         """
#         target_embedding = self.node_embeddings[target_node]
#         context_embedding = self.node_embeddings[context_node]

#         if negative:
#             loss = -np.log(1 - np.sum(target_embedding * context_embedding))
#         else:
#             loss = -np.log(sigmoid(np.sum(target_embedding * context_embedding)))

#         return loss
#     def skipgram_loss_1(self, target_node, context_node, negative=False):
#         """
#         Calculate the loss for skip-gram with negative sampling.

#         Args:
#             target_node: The embedding of the target node.
#             context_node: The embedding of the context node (positive or negative).
#             negative (bool): Whether it's a negative sample (default: False).

#         Returns:
#             float: The loss.
#         """
#         if target_node not in self.node_embeddings or context_node not in self.node_embeddings:
#             return 0.0  # Return a default value or handle this case appropriately

#         target_embedding = self.node_embeddings[target_node]
#         context_embedding = self.node_embeddings[context_node]

#         if negative:
#             loss = -np.log(1 - np.sum(target_embedding * context_embedding))
#         else:
#             loss = -np.log(sigmoid(np.sum(target_embedding * context_embedding)))

#         return loss

    def skipgram_loss(self, target_node, context_node, negative=False):
        """
        Calculate the loss for skip-gram with negative sampling.

        Args:
            target_node: The embedding of the target node.
            context_node: The embedding of the context node (positive or negative).
            negative (bool): Whether it's a negative sample (default: False).

        Returns:
            float: The loss.
        """
        if target_node not in self.node_embeddings or context_node not in self.node_embeddings:
            return 0.0  # Return a default value or handle this case appropriately

        target_embedding = self.node_embeddings[target_node]
        context_embedding = self.node_embeddings[context_node]

        if negative:
            loss = -np.log(1 - sigmoid(np.sum(target_embedding * context_embedding)))
        else:
            loss = -np.log(sigmoid(np.sum(target_embedding * context_embedding)))

        return loss



    def get_embeddings(self):
        """
        Get node embeddings for all nodes in the graph.

        Returns:
            dict: A dictionary mapping nodes to their embeddings.
        """
        return {node: self.node_embeddings[node] for node in self.graph.nodes()}

    def plot_loss(self, loss_history):
        """
        Save and display the training loss plot.

        Args:
            loss_history (list): List of loss values for each epoch.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.epochs + 1), loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)

        save_folder = "loss_plots"
        os.makedirs(save_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_p{self.p}_q{self.q}_walk{self.walk_length}_win{self.window_size}.png"
        filepath = os.path.join(save_folder, filename)

        plt.savefig(filepath)
        plt.show()
        plt.close()

# Define the sigmoid function (NumPy equivalent)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
