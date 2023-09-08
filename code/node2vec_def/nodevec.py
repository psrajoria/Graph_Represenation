# # # import os
# # # import networkx as nx
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # from concurrent.futures import ProcessPoolExecutor
# # # from datetime import datetime
# # # from tqdm import tqdm
# # # import time

# # # # Define the sigmoid function (NumPy equivalent)
# # # def sigmoid(x):
# # #     """
# # #     Calculate the sigmoid of a given value.

# # #     Args:
# # #         x (float): Input value.

# # #     Returns:
# # #         float: Sigmoid of the input value.
# # #     """
# # #     return 1 / (1 + np.exp(-x))

# # # class CustomTQDM(tqdm):
# # #     def __init__(self, total=None, desc='', leave=True, dynamic_ncols=True):
# # #         """
# # #         Custom progress bar based on tqdm.

# # #         Args:
# # #             total (int, optional): Total number of iterations. Defaults to None.
# # #             desc (str, optional): Description to display. Defaults to ''.
# # #             leave (bool, optional): Whether to leave the progress bar after completion. Defaults to True.
# # #             dynamic_ncols (bool, optional): Whether to allow dynamic resizing of the progress bar. Defaults to True.
# # #         """
# # #         super().__init__(total=total, desc=desc, leave=leave, dynamic_ncols=dynamic_ncols)
# # #         self.epoch_start_time = None

# # #     def set_epoch_start_time(self):
# # #         """
# # #         Set the start time of the current epoch.
# # #         """
# # #         self.epoch_start_time = time.time()

# # #     def update_with_epoch_time(self):
# # #         """
# # #         Update the progress bar with epoch time.

# # #         Calculates and displays the time taken for the current epoch.
# # #         """
# # #         if self.epoch_start_time is not None:
# # #             epoch_time = time.time() - self.epoch_start_time
# # #             self.set_description(f'{self.desc} | Epoch Time: {epoch_time:.2f}s')
# # #         self.update(1)

# # # class Node2Vec:
# # #     """
# # #     Node2Vec algorithm for learning node embeddings in a graph.
# # #     (Modified to use CPU and NumPy)

# # #     Args:
# # #         graph (nx.Graph): The input graph.
# # #         dimensions (int): The dimensionality of the node embeddings.
# # #         walk_length (int): Length of each random walk.
# # #         num_walks (int): Number of random walks to perform per node.
# # #         p (float): Return parameter for controlling BFS exploration.
# # #         q (float): In-out parameter for controlling BFS exploration.
# # #         T (int): Length of random walks for similarity.
# # #         learning_rate (float): Learning rate for stochastic gradient descent.
# # #         window_size (int): Maximum distance between the current and predicted node within a sentence.
# # #         epochs (int): Number of training epochs.
# # #         negative_samples (int): Number of negative samples per positive pair.

# # #     Attributes:
# # #         graph (nx.Graph): The input graph.
# # #         dimensions (int): The dimensionality of the node embeddings.
# # #         walk_length (int): Length of each random walk.
# # #         num_walks (int): Number of random walks to perform per node.
# # #         p (float): Return parameter for controlling BFS exploration.
# # #         q (float): In-out parameter for controlling BFS exploration.
# # #         T (int): Length of random walks for similarity.
# # #         learning_rate (float): Learning rate for stochastic gradient descent.
# # #         window_size (int): Maximum distance between the current and predicted node within a sentence.
# # #         epochs (int): Number of training epochs.
# # #         negative_samples (int): Number of negative samples per positive pair.
# # #         device (str): The device ('cpu') for computation.
# # #         node_embeddings (numpy.ndarray): Node embeddings model.
# # #         transition_probs (dict): Precomputed transition probabilities for random walks.

# # #     Methods:
# # #         compute_transition_probs(): Compute transition probabilities for each node in the graph.
# # #         compute_transition_probs_single(current_node, neighbors): Compute transition probabilities for a single node.
# # #         node2vec_walk(start_node): Generate a single biased random walk starting from the given node.
# # #         node2vec_step(current_node, previous_node, neighbors, transition_probs):
# # #             Compute the next step in a biased random walk.
# # #         train(): Train the Node2Vec model.
# # #         update_embeddings(walk): Update node embeddings based on random walks.
# # #         skipgram_loss(target_node, context_node, negative=False):
# # #             Calculate the loss for skip-gram with negative sampling.
# # #         get_embeddings(): Get node embeddings for all nodes in the graph.
# # #         plot_loss(loss_history): Save and display the training loss plot.
# # #     """

# # #     def __init__(self, graph, dimensions, walk_length, num_walks, p, q, T, learning_rate, window_size, epochs,
# # #                  negative_samples):
# # #         """
# # #         Initialize the Node2Vec model.

# # #         Args:
# # #             graph (nx.Graph): The input graph.
# # #             dimensions (int): The dimensionality of the node embeddings.
# # #             walk_length (int): Length of each random walk.
# # #             num_walks (int): Number of random walks to perform per node.
# # #             p (float): Return parameter for controlling BFS exploration.
# # #             q (float): In-out parameter for controlling BFS exploration.
# # #             T (int): Length of random walks for similarity.
# # #             learning_rate (float): Learning rate for stochastic gradient descent.
# # #             window_size (int): Maximum distance between the current and predicted node within a sentence.
# # #             epochs (int): Number of training epochs.
# # #             negative_samples (int): Number of negative samples per positive pair.
# # #         """
# # #         self.graph = graph.to_directed()
# # #         self.dimensions = dimensions
# # #         self.walk_length = walk_length
# # #         self.num_walks = num_walks
# # #         self.p = p
# # #         self.q = q
# # #         self.T = T
# # #         self.learning_rate = learning_rate
# # #         self.window_size = window_size
# # #         self.epochs = epochs
# # #         self.negative_samples = negative_samples

# # #         # Set the device to CPU
# # #         self.device = 'cpu'

# # #         # Initialize node embeddings (using NumPy)
# # #         self.node_embeddings = np.random.rand(len(graph.nodes()), dimensions)

# # #         # Precompute transition probabilities for efficient random walks
# # #         self.transition_probs = self.compute_transition_probs()

# # #     def compute_transition_probs(self):
# # #         """
# # #         Compute transition probabilities for each node in the graph.

# # #         Returns:
# # #             dict: A dictionary mapping nodes to their transition probabilities.
# # #         """
# # #         transition_probs = {}
# # #         for node in self.graph.nodes():
# # #             neighbors = list(self.graph.neighbors(node))
# # #             probs = self.compute_transition_probs_single(node, neighbors)
# # #             transition_probs[node] = (neighbors, probs)
# # #         return transition_probs

# # #     def compute_transition_probs_single(self, current_node, neighbors):
# # #         """
# # #         Compute transition probabilities for a single node.

# # #         Args:
# # #             current_node: The current node in the random walk.
# # #             neighbors: List of neighboring nodes.

# # #         Returns:
# # #             list: List of transition probabilities for neighboring nodes.
# # #         """
# # #         probs = []
# # #         for neighbor in neighbors:
# # #             if neighbor == current_node:
# # #                 probs.append(1 / self.p)
# # #             elif neighbor in self.graph[current_node]:
# # #                 probs.append(1)
# # #             else:
# # #                 probs.append(1 / self.q)

# # #         normalized_probs = np.array(probs) / np.sum(probs)
# # #         return normalized_probs

# # #     def node2vec_walk(self, start_node):
# # #         """
# # #         Generate a single biased random walk starting from the given node.

# # #         Args:
# # #             start_node: The starting node for the random walk.

# # #         Returns:
# # #             list: A list representing the generated random walk.
# # #         """
# # #         walk = [start_node]
# # #         while len(walk) < self.walk_length:
# # #             current_node = walk[-1]
# # #             neighbors, transition_probs = self.transition_probs[current_node]
# # #             if len(neighbors) > 0:
# # #                 next_node = self.node2vec_step(current_node, walk[-2] if len(walk) > 1 else None, neighbors,
# # #                                                transition_probs)
# # #                 walk.append(next_node)
# # #             else:
# # #                 break
# # #         return walk

# # #     def node2vec_step(self, current_node, previous_node, neighbors, transition_probs):
# # #         """
# # #         Compute the next step in a biased random walk.

# # #         Args:
# # #             current_node: The current node in the random walk.
# # #             previous_node: The previous node in the random walk.
# # #             neighbors: List of neighboring nodes.
# # #             transition_probs: List of transition probabilities for neighboring nodes.

# # #         Returns:
# # #             int: The next node in the random walk.
# # #         """
# # #         if previous_node is None:
# # #             rand_index = np.random.choice(len(neighbors))
# # #             return neighbors[rand_index]

# # #         rand_index = np.random.choice(len(neighbors))
# # #         return neighbors[rand_index]

# # #     def train(self):
# # #         """
# # #         Train the Node2Vec model.

# # #         This method performs the training process, which includes generating random walks, updating node embeddings,
# # #         and tracking loss.

# # #         Returns:
# # #             None
# # #         """
# # #         loss_history = []

# # #         for epoch in range(self.epochs):
# # #             total_loss = 0.0
# # #             epoch_bar = CustomTQDM(total=self.num_walks * len(self.graph.nodes()), desc=f"Epoch {epoch + 1}/{self.epochs}",
# # #                         leave=False, dynamic_ncols=True)

# # #             with ProcessPoolExecutor() as executor:
# # #                 futures = []
# # #                 for _ in range(self.num_walks):
# # #                     for node in self.graph.nodes():
# # #                         futures.append(executor.submit(self.node2vec_walk, node))

# # #                 for future in futures:
# # #                     walk = future.result()
# # #                     loss = self.update_embeddings(walk, epoch_bar)
# # #                     total_loss += loss

# # #                 epoch_bar.close()
# # #                 loss_history.append(total_loss / (self.num_walks * len(self.graph.nodes())))

# # #         self.plot_loss(loss_history)

# # #     def update_embeddings(self, walk, epoch_bar):
# # #         """
# # #         Update node embeddings based on random walks.

# # #         Args:
# # #             walk (list): The random walk sequence.
# # #             epoch_bar (CustomTQDM): The progress bar for the current epoch.

# # #         Returns:
# # #             float: The total loss for the walk.
# # #         """
# # #         total_loss = 0.0

# # #         for i, target_node in enumerate(walk):
# # #             positive_samples = [walk[j] for j in
# # #                                 range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)) if
# # #                                 i != j]
# # #             for context_node in positive_samples:
# # #                 loss = self.skipgram_loss(target_node, context_node)
# # #                 total_loss += loss

# # #             negative_samples = np.random.choice(len(self.graph.nodes()), size=self.negative_samples)
# # #             for negative_context_node in negative_samples:
# # #                 loss = self.skipgram_loss(target_node, negative_context_node, negative=True)
# # #                 total_loss += loss

# # #         # Update the epoch's progress bar with the current step
# # #         epoch_bar.update(1)

# # #         return total_loss

# # #     def skipgram_loss(self, target_node, context_node, negative=False):
# # #         """
# # #         Calculate the loss for skip-gram with negative sampling.

# # #         Args:
# # #             target_node: The embedding of the target node.
# # #             context_node: The embedding of the context node (positive or negative).
# # #             negative (bool): Whether it's a negative sample (default: False).

# # #         Returns:
# # #             float: The loss.
# # #         """
# # #         if target_node not in self.node_embeddings or context_node not in self.node_embeddings:
# # #             return 0.0  # Return a default value or handle this case appropriately

# # #         target_embedding = self.node_embeddings[target_node]
# # #         context_embedding = self.node_embeddings[context_node]

# # #         if negative:
# # #             loss = -np.log(1 - sigmoid(np.sum(target_embedding * context_embedding)))
# # #         else:
# # #             loss = -np.log(sigmoid(np.sum(target_embedding * context_embedding)))

# # #         return loss

# # #     def get_embeddings(self):
# # #         """
# # #         Get node embeddings for all nodes in the graph.

# # #         Returns:
# # #             dict: A dictionary mapping nodes to their embeddings.
# # #         """
# # #         return {node: self.node_embeddings[node] for node in self.graph.nodes()}

# # #     def plot_loss(self, loss_history):
# # #         """
# # #         Save and display the training loss plot.

# # #         Args:
# # #             loss_history (list): List of loss values for each epoch.
# # #         """
# # #         plt.figure(figsize=(10, 5))
# # #         plt.plot(range(1, self.epochs + 1), loss_history)
# # #         plt.xlabel('Epoch')
# # #         plt.ylabel('Loss')
# # #         plt.title('Training Loss')
# # #         plt.grid(True)

# # #         save_folder = "loss_plots"
# # #         os.makedirs(save_folder, exist_ok=True)

# # #         timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
# # #         filename = f"{timestamp}_p{self.p}_q{self.q}_walk{self.walk_length}_win{self.window_size}.png"
# # #         filepath = os.path.join(save_folder, filename)

# # #         plt.savefig(filepath)
# # #         plt.show()
# # #         plt.close()


# # import os
# # import networkx as nx
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from concurrent.futures import ProcessPoolExecutor
# # from datetime import datetime
# # from tqdm import tqdm
# # import time

# # # Define the sigmoid function (NumPy equivalent)
# # def sigmoid(x):
# #     """
# #     Calculate the sigmoid of a given value.

# #     Args:
# #         x (float): Input value.

# #     Returns:
# #         float: Sigmoid of the input value.
# #     """
# #     return 1 / (1 + np.exp(-x))

# # class CustomTQDM(tqdm):
# #     def __init__(self, total=None, desc='', leave=True, dynamic_ncols=True):
# #         """
# #         Custom progress bar based on tqdm.

# #         Args:
# #             total (int, optional): Total number of iterations. Defaults to None.
# #             desc (str, optional): Description to display. Defaults to ''.
# #             leave (bool, optional): Whether to leave the progress bar after completion. Defaults to True.
# #             dynamic_ncols (bool, optional): Whether to allow dynamic resizing of the progress bar. Defaults to True.
# #         """
# #         super().__init__(total=total, desc=desc, leave=leave, dynamic_ncols=dynamic_ncols)
# #         self.epoch_start_time = None

# #     def set_epoch_start_time(self):
# #         """
# #         Set the start time of the current epoch.
# #         """
# #         self.epoch_start_time = time.time()

# #     def update_with_epoch_time(self):
# #         """
# #         Update the progress bar with epoch time.

# #         Calculates and displays the time taken for the current epoch.
# #         """
# #         if self.epoch_start_time is not None:
# #             epoch_time = time.time() - self.epoch_start_time
# #             self.set_description(f'{self.desc} | Epoch Time: {epoch_time:.2f}s')
# #         self.update(1)

# # class Node2Vec:
# #     """
# #     Node2Vec algorithm for learning node embeddings in a graph.
# #     (Modified to use CPU and NumPy)

# #     Args:
# #         graph (nx.Graph): The input graph.
# #         dimensions (int): The dimensionality of the node embeddings.
# #         walk_length (int): Length of each random walk.
# #         num_walks (int): Number of random walks to perform per node.
# #         p (float): Return parameter for controlling BFS exploration.
# #         q (float): In-out parameter for controlling BFS exploration.
# #         T (int): Length of random walks for similarity.
# #         learning_rate (float): Learning rate for stochastic gradient descent.
# #         window_size (int): Maximum distance between the current and predicted node within a sentence.
# #         epochs (int): Number of training epochs.
# #         negative_samples (int): Number of negative samples per positive pair.

# #     Attributes:
# #         graph (nx.Graph): The input graph.
# #         dimensions (int): The dimensionality of the node embeddings.
# #         walk_length (int): Length of each random walk.
# #         num_walks (int): Number of random walks to perform per node.
# #         p (float): Return parameter for controlling BFS exploration.
# #         q (float): In-out parameter for controlling BFS exploration.
# #         T (int): Length of random walks for similarity.
# #         learning_rate (float): Learning rate for stochastic gradient descent.
# #         window_size (int): Maximum distance between the current and predicted node within a sentence.
# #         epochs (int): Number of training epochs.
# #         negative_samples (int): Number of negative samples per positive pair.
# #         device (str): The device ('cpu') for computation.
# #         node_embeddings (numpy.ndarray): Node embeddings model.
# #         transition_probs (dict): Precomputed transition probabilities for random walks.

# #     Methods:
# #         compute_transition_probs(): Compute transition probabilities for each node in the graph.
# #         compute_transition_probs_single(current_node, neighbors): Compute transition probabilities for a single node.
# #         node2vec_walk(start_node): Generate a single biased random walk starting from the given node.
# #         node2vec_step(current_node, previous_node, neighbors, transition_probs, negative=False):
# #             Compute the next step in a biased random walk and update node embeddings.
# #         train(): Train the Node2Vec model.
# #         update_embeddings(walk, epoch_bar): Update node embeddings based on random walks.
# #         skipgram_loss(target_node, context_node, negative=False):
# #             Calculate the loss for skip-gram with negative sampling.
# #         get_embeddings(): Get node embeddings for all nodes in the graph.
# #         plot_loss(loss_history): Save and display the training loss plot.
# #     """

# #     def __init__(self, graph, dimensions, walk_length, num_walks, p, q, T, learning_rate, window_size, epochs,
# #                  negative_samples):
# #         """
# #         Initialize the Node2Vec model.

# #         Args:
# #             graph (nx.Graph): The input graph.
# #             dimensions (int): The dimensionality of the node embeddings.
# #             walk_length (int): Length of each random walk.
# #             num_walks (int): Number of random walks to perform per node.
# #             p (float): Return parameter for controlling BFS exploration.
# #             q (float): In-out parameter for controlling BFS exploration.
# #             T (int): Length of random walks for similarity.
# #             learning_rate (float): Learning rate for stochastic gradient descent.
# #             window_size (int): Maximum distance between the current and predicted node within a sentence.
# #             epochs (int): Number of training epochs.
# #             negative_samples (int): Number of negative samples per positive pair.
# #         """
# #         self.graph = graph.to_directed()
# #         self.dimensions = dimensions
# #         self.walk_length = walk_length
# #         self.num_walks = num_walks
# #         self.p = p
# #         self.q = q
# #         self.T = T
# #         self.learning_rate = learning_rate
# #         self.window_size = window_size
# #         self.epochs = epochs
# #         self.negative_samples = negative_samples

# #         # Set the device to CPU
# #         self.device = 'cpu'

# #         # Initialize node embeddings (using NumPy)
# #         self.node_embeddings = np.random.rand(len(graph.nodes()), dimensions)

# #         # Precompute transition probabilities for efficient random walks
# #         self.transition_probs = self.compute_transition_probs()

# #     def compute_transition_probs(self):
# #         """
# #         Compute transition probabilities for each node in the graph.

# #         Returns:
# #             dict: A dictionary mapping nodes to their transition probabilities.
# #         """
# #         transition_probs = {}
# #         for node in self.graph.nodes():
# #             neighbors = list(self.graph.neighbors(node))
# #             probs = self.compute_transition_probs_single(node, neighbors)
# #             transition_probs[node] = (neighbors, probs)
# #         return transition_probs

# #     def compute_transition_probs_single(self, current_node, neighbors):
# #         """
# #         Compute transition probabilities for a single node.

# #         Args:
# #             current_node: The current node in the random walk.
# #             neighbors: List of neighboring nodes.

# #         Returns:
# #             list: List of transition probabilities for neighboring nodes.
# #         """
# #         probs = []
# #         for neighbor in neighbors:
# #             if neighbor == current_node:
# #                 probs.append(1 / self.p)
# #             elif neighbor in self.graph[current_node]:
# #                 probs.append(1)
# #             else:
# #                 probs.append(1 / self.q)

# #         normalized_probs = np.array(probs) / np.sum(probs)
# #         return normalized_probs

# #     def node2vec_walk(self, start_node):
# #         """
# #         Generate a single biased random walk starting from the given node.

# #         Args:
# #             start_node: The starting node for the random walk.

# #         Returns:
# #             list: A list representing the generated random walk.
# #         """
# #         walk = [start_node]
# #         while len(walk) < self.walk_length:
# #             current_node = walk[-1]
# #             neighbors, transition_probs = self.transition_probs[current_node]
# #             if len(neighbors) > 0:
# #                 next_node = self.node2vec_step(current_node, walk[-2] if len(walk) > 1 else None, neighbors,
# #                                                transition_probs)
# #                 walk.append(next_node)
# #             else:
# #                 break
# #         return walk

# #     def node2vec_step(self, current_node, previous_node, neighbors, transition_probs, negative=False):
# #         """
# #         Compute the next step in a biased random walk and update node embeddings.

# #         Args:
# #             current_node: The current node in the random walk.
# #             previous_node: The previous node in the random walk.
# #             neighbors: List of neighboring nodes.
# #             transition_probs: List of transition probabilities for neighboring nodes.
# #             negative (bool, optional): Whether it's a negative sample (default: False).

# #         Returns:
# #             int: The next node in the random walk.
# #         """
# #         if previous_node is None:
# #             rand_index = np.random.choice(len(neighbors))
# #             next_node = neighbors[rand_index]
# #         else:
# #             normalized_probs = transition_probs
# #             rand_index = np.random.choice(len(neighbors), p=normalized_probs)
# #             next_node = neighbors[rand_index]

# #         # Update the node embeddings using gradient descent
# #         target_embedding = self.node_embeddings[current_node]
# #         context_embedding = self.node_embeddings[next_node]
# #         error = sigmoid(np.sum(target_embedding * context_embedding))
        
# #         # Compute the gradient
# #         grad = error - 1 if not negative else error

# #         # Update the embeddings
# #         self.node_embeddings[current_node] -= self.learning_rate * grad * context_embedding
# #         self.node_embeddings[next_node] -= self.learning_rate * grad * target_embedding

# #         return next_node

# #     def train(self):
# #         """
# #         Train the Node2Vec model.

# #         This method performs the training process, which includes generating random walks, updating node embeddings,
# #         and tracking loss.

# #         Returns:
# #             None
# #         """
# #         loss_history = []

# #         for epoch in range(self.epochs):
# #             total_loss = 0.0
# #             epoch_bar = CustomTQDM(total=self.num_walks * len(self.graph.nodes()), desc=f"Epoch {epoch + 1}/{self.epochs}",
# #                         leave=False, dynamic_ncols=True)

# #             with ProcessPoolExecutor() as executor:
# #                 futures = []
# #                 for _ in range(self.num_walks):
# #                     for node in self.graph.nodes():
# #                         futures.append(executor.submit(self.node2vec_walk, node))

# #                 for future in futures:
# #                     walk = future.result()
# #                     loss = self.update_embeddings(walk, epoch_bar)
# #                     total_loss += loss

# #                 epoch_bar.close()
# #                 loss_history.append(total_loss / (self.num_walks * len(self.graph.nodes())))

# #         self.plot_loss(loss_history)

# #     def update_embeddings(self, walk, epoch_bar):
# #         """
# #         Update node embeddings based on random walks.

# #         Args:
# #             walk (list): The random walk sequence.
# #             epoch_bar (CustomTQDM): The progress bar for the current epoch.

# #         Returns:
# #             float: The total loss for the walk.
# #         """
# #         total_loss = 0.0

# #         for i, target_node in enumerate(walk):
# #             positive_samples = [walk[j] for j in
# #                                 range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)) if
# #                                 i != j]
# #             for context_node in positive_samples:
# #                 loss = self.skipgram_loss(target_node, context_node)
# #                 total_loss += loss

# #             negative_samples = np.random.choice(len(self.graph.nodes()), size=self.negative_samples)
# #             for negative_context_node in negative_samples:
# #                 loss = self.skipgram_loss(target_node, negative_context_node, negative=True)
# #                 total_loss += loss

# #         # Update the epoch's progress bar with the current step
# #         epoch_bar.update(1)

# #         return total_loss

# #     def skipgram_loss(self, target_node, context_node, negative=False):
# #         """
# #         Calculate the loss for skip-gram with negative sampling.

# #         Args:
# #             target_node: The embedding of the target node.
# #             context_node: The embedding of the context node (positive or negative).
# #             negative (bool, optional): Whether it's a negative sample (default: False).

# #         Returns:
# #             float: The loss.
# #         """
# #         if target_node not in self.node_embeddings or context_node not in self.node_embeddings:
# #             return 0.0  # Return a default value or handle this case appropriately

# #         target_embedding = self.node_embeddings[target_node]
# #         context_embedding = self.node_embeddings[context_node]

# #         if negative:
# #             loss = -np.log(1 - sigmoid(np.sum(target_embedding * context_embedding)))
# #         else:
# #             loss = -np.log(sigmoid(np.sum(target_embedding * context_embedding)))

# #         return loss

# #     def get_embeddings(self):
# #         """
# #         Get node embeddings for all nodes in the graph.

# #         Returns:
# #             dict: A dictionary mapping nodes to their embeddings.
# #         """
# #         return {node: self.node_embeddings[node] for node in self.graph.nodes()}

# #     def plot_loss(self, loss_history):
# #         """
# #         Save and display the training loss plot.

# #         Args:
# #             loss_history (list): List of loss values for each epoch.
# #         """
# #         plt.figure(figsize=(10, 5))
# #         plt.plot(range(1, self.epochs + 1), loss_history)
# #         plt.xlabel('Epoch')
# #         plt.ylabel('Loss')
# #         plt.title('Training Loss')
# #         plt.grid(True)

# #         save_folder = "loss_plots"
# #         os.makedirs(save_folder, exist_ok=True)

# #         timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
# #         filename = f"{timestamp}_p{self.p}_q{self.q}_walk{self.walk_length}_win{self.window_size}.png"
# #         filepath = os.path.join(save_folder, filename)

# #         plt.savefig(filepath)
# #         plt.show()
# #         plt.close()


# import os
# import networkx as nx
# import numpy as np
# import matplotlib.pyplot as plt
# from concurrent.futures import ProcessPoolExecutor
# from datetime import datetime
# from tqdm import tqdm
# import time

# # Define the sigmoid function (NumPy equivalent)
# def sigmoid(x):
#     """
#     Calculate the sigmoid of a given value.

#     Args:
#         x (float): Input value.

#     Returns:
#         float: Sigmoid of the input value.
#     """
#     return 1 / (1 + np.exp(-x))

# class CustomTQDM(tqdm):
#     def __init__(self, total=None, desc='', leave=True, dynamic_ncols=True):
#         """
#         Custom progress bar based on tqdm.

#         Args:
#             total (int, optional): Total number of iterations. Defaults to None.
#             desc (str, optional): Description to display. Defaults to ''.
#             leave (bool, optional): Whether to leave the progress bar after completion. Defaults to True.
#             dynamic_ncols (bool, optional): Whether to allow dynamic resizing of the progress bar. Defaults to True.
#         """
#         super().__init__(total=total, desc=desc, leave=leave, dynamic_ncols=dynamic_ncols)
#         self.epoch_start_time = None

#     def set_epoch_start_time(self):
#         """
#         Set the start time of the current epoch.
#         """
#         self.epoch_start_time = time.time()

#     def update_with_epoch_time(self):
#         """
#         Update the progress bar with epoch time.

#         Calculates and displays the time taken for the current epoch.
#         """
#         if self.epoch_start_time is not None:
#             epoch_time = time.time() - self.epoch_start_time
#             self.set_description(f'{self.desc} | Epoch Time: {epoch_time:.2f}s')
#         self.update(1)

# class Node2Vec:
#     """
#     Node2Vec algorithm for learning node embeddings in a graph.
#     (Modified to use CPU and NumPy)

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
#         device (str): The device ('cpu') for computation.
#         node_embeddings (numpy.ndarray): Node embeddings model.
#         transition_probs (dict): Precomputed transition probabilities for random walks.

#     Methods:
#         compute_transition_probs(): Compute transition probabilities for each node in the graph.
#         compute_transition_probs_single(current_node, neighbors): Compute transition probabilities for a single node.
#         node2vec_walk(start_node): Generate a single biased random walk starting from the given node.
#         node2vec_step(current_node, context_node, negative=False):
#             Compute the next step in a biased random walk and update node embeddings.
#         train(): Train the Node2Vec model.
#         update_embeddings(walk, epoch_bar): Update node embeddings based on random walks.
#         skipgram_loss(target_node, context_node, negative=False):
#             Calculate the loss for skip-gram with negative sampling.
#         get_embeddings(): Get node embeddings for all nodes in the graph.
#         plot_loss(loss_history): Save and display the training loss plot.
#     """

#     def __init__(self, graph, dimensions, walk_length, num_walks, p, q, T, learning_rate, window_size, epochs,
#                  negative_samples):
#         """
#         Initialize the Node2Vec model.

#         Args:
#             graph (nx.Graph): The input graph.
#             dimensions (int): The dimensionality of the node embeddings.
#             walk_length (int): Length of each random walk.
#             num_walks (int): Number of random walks to perform per node.
#             p (float): Return parameter for controlling BFS exploration.
#             q (float): In-out parameter for controlling BFS exploration.
#             T (int): Length of random walks for similarity.
#             learning_rate (float): Learning rate for stochastic gradient descent.
#             window_size (int): Maximum distance between the current and predicted node within a sentence.
#             epochs (int): Number of training epochs.
#             negative_samples (int): Number of negative samples per positive pair.
#         """
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

#         # Set the device to CPU
#         self.device = 'cpu'

#         # Initialize node embeddings (using NumPy)
#         self.node_embeddings = np.random.rand(len(graph.nodes()), dimensions)

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

#         normalized_probs = np.array(probs) / np.sum(probs)
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

#     def node2vec_step(self, current_node, previous_node, neighbors):
#         """
#         Compute the next step in a biased random walk and update node embeddings.

#         Args:
#             current_node: The current node in the random walk.
#             previous_node: The previous node in the random walk.
#             neighbors: List of neighboring nodes.

#         Returns:
#             int: The next node in the random walk.
#         """
#         if previous_node is None:
#             rand_index = np.random.choice(len(neighbors))
#             next_node = neighbors[rand_index]
#         else:
#             normalized_probs = self.transition_probs[current_node][1]
#             rand_index = np.random.choice(len(neighbors), p=normalized_probs)
#             next_node = neighbors[rand_index]

#         # Update the node embeddings using gradient descent
#         target_embedding = self.node_embeddings[current_node]
#         context_embedding = self.node_embeddings[next_node]
#         error = sigmoid(np.sum(target_embedding * context_embedding))

#         # Compute the gradient
#         grad = error - 1

#         # Update the embeddings
#         self.node_embeddings[current_node] -= self.learning_rate * grad * context_embedding
#         self.node_embeddings[next_node] -= self.learning_rate * grad * target_embedding

#         return next_node

#     def train(self):
#         """
#         Train the Node2Vec model.

#         This method performs the training process, which includes generating random walks, updating node embeddings,
#         and tracking loss.

#         Returns:
#             None
#         """
#         loss_history = []

#         for epoch in range(self.epochs):
#             total_loss = 0.0
#             epoch_bar = CustomTQDM(total=self.num_walks * len(self.graph.nodes()), desc=f"Epoch {epoch + 1}/{self.epochs}",
#                         leave=False, dynamic_ncols=True)

#             with ProcessPoolExecutor() as executor:
#                 futures = []
#                 for _ in range(self.num_walks):
#                     for node in self.graph.nodes():
#                         futures.append(executor.submit(self.node2vec_walk, node))

#                 for future in futures:
#                     walk = future.result()
#                     loss = self.update_embeddings(walk, epoch_bar)
#                     total_loss += loss

#                 epoch_bar.close()
#                 loss_history.append(total_loss / (self.num_walks * len(self.graph.nodes())))

#         self.plot_loss(loss_history)

#     def update_embeddings(self, walk, epoch_bar):
#         """
#         Update node embeddings based on random walks.

#         Args:
#             walk (list): The random walk sequence.
#             epoch_bar (CustomTQDM): The progress bar for the current epoch.

#         Returns:
#             float: The total loss for the walk.
#         """
#         total_loss = 0.0

#         for i, target_node in enumerate(walk):
#             positive_samples = [walk[j] for j in
#                                 range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)) if
#                                 i != j]
#             for context_node in positive_samples:
#                 loss = self.skipgram_loss(target_node, context_node)
#                 total_loss += loss

#                 # Update the embeddings
#                 self.node2vec_step(target_node, context_node)

#             negative_samples = np.random.choice(len(self.graph.nodes()), size=self.negative_samples)
#             for negative_context_node in negative_samples:
#                 loss = self.skipgram_loss(target_node, negative_context_node, negative=True)
#                 total_loss += loss

#                 # Update the embeddings
#                 self.node2vec_step(target_node, negative_context_node, negative=True)

#         # Update the epoch's progress bar with the current step
#         epoch_bar.update(1)

#         return total_loss

#     def skipgram_loss(self, target_node, context_node, negative=False):
#         """
#         Calculate the loss for skip-gram with negative sampling.

#         Args:
#             target_node: The embedding of the target node.
#             context_node: The embedding of the context node (positive or negative).
#             negative (bool, optional): Whether it's a negative sample (default: False).

#         Returns:
#             float: The loss.
#         """
#         if target_node not in self.node_embeddings or context_node not in self.node_embeddings:
#             return 0.0  # Return a default value or handle this case appropriately

#         target_embedding = self.node_embeddings[target_node]
#         context_embedding = self.node_embeddings[context_node]

#         if negative:
#             loss = -np.log(1 - sigmoid(np.sum(target_embedding * context_embedding)))
#         else:
#             loss = -np.log(sigmoid(np.sum(target_embedding * context_embedding)))

#         return loss

#     def get_embeddings(self):
#         """
#         Get node embeddings for all nodes in the graph.

#         Returns:
#             dict: A dictionary mapping nodes to their embeddings.
#         """
#         return {node: self.node_embeddings[node] for node in self.graph.nodes()}

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



#### working
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from tqdm import tqdm
import time

# Define the sigmoid function (NumPy equivalent)
def sigmoid(x):
    """
    Calculate the sigmoid of a given value.

    Args:
        x (float): Input value.

    Returns:
        float: Sigmoid of the input value.
    """
    return 1 / (1 + np.exp(-x))

class CustomTQDM(tqdm):
    def __init__(self, total=None, desc='', leave=True, dynamic_ncols=True):
        """
        Custom progress bar based on tqdm.

        Args:
            total (int, optional): Total number of iterations. Defaults to None.
            desc (str, optional): Description to display. Defaults to ''.
            leave (bool, optional): Whether to leave the progress bar after completion. Defaults to True.
            dynamic_ncols (bool, optional): Whether to allow dynamic resizing of the progress bar. Defaults to True.
        """
        super().__init__(total=total, desc=desc, leave=leave, dynamic_ncols=dynamic_ncols)
        self.epoch_start_time = None

    def set_epoch_start_time(self):
        """
        Set the start time of the current epoch.
        """
        self.epoch_start_time = time.time()

    def update_with_epoch_time(self):
        """
        Update the progress bar with epoch time.

        Calculates and displays the time taken for the current epoch.
        """
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.set_description(f'{self.desc} | Epoch Time: {epoch_time:.2f}s')
        self.update(1)

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
        node2vec_step(current_node, previous_node, neighbors, negative=False):
            Compute the next step in a biased random walk and update node embeddings.
        train(): Train the Node2Vec model.
        update_embeddings(walk, epoch_bar): Update node embeddings based on random walks.
        skipgram_loss(target_node, context_node, negative=False):
            Calculate the loss for skip-gram with negative sampling.
        get_embeddings(): Get node embeddings for all nodes in the graph.
        plot_loss(loss_history): Save and display the training loss plot.
    """

    def __init__(self, graph, dimensions, walk_length, num_walks, p, q, T, learning_rate, window_size, epochs,
                 negative_samples):
        """
        Initialize the Node2Vec model.

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
        """
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
                next_node = self.node2vec_step(current_node, walk[-2] if len(walk) > 1 else None, neighbors)
                walk.append(next_node)
            else:
                break
        return walk

#     def node2vec_step(self, current_node, previous_node, neighbors, negative=False):
#         """
#         Compute the next step in a biased random walk and update node embeddings.

#         Args:
#             current_node: The current node in the random walk.
#             previous_node: The previous node in the random walk.
#             neighbors: List of neighboring nodes.
#             negative (bool, optional): Whether it's a negative sample (default: False).

#         Returns:
#             int: The next node in the random walk.
#         """
#         if previous_node is None:
#             rand_index = np.random.choice(len(neighbors))
#             next_node = neighbors[rand_index]
#         else:
#             normalized_probs = self.transition_probs[current_node][1]
#             rand_index = np.random.choice(len(neighbors), p=normalized_probs)
#             next_node = neighbors[rand_index]

#         # Update the node embeddings using gradient descent
#         target_embedding = self.node_embeddings[current_node]
#         context_embedding = self.node_embeddings[next_node]
#         error = sigmoid(np.sum(target_embedding * context_embedding))

#         # Compute the gradient
#         grad = error - 1

#         # Update the embeddings
#         self.node_embeddings[current_node] -= self.learning_rate * grad * context_embedding
#         self.node_embeddings[next_node] -= self.learning_rate * grad * target_embedding

#         return next_node
    def node2vec_step(self, current_node, previous_node, neighbors, negative=False):
        """
        Compute the next step in a biased random walk and update node embeddings.

        Args:
            current_node: The current node in the random walk.
            previous_node: The previous node in the random walk.
            neighbors: List of neighboring nodes.
            negative (bool, optional): Whether it's a negative sample (default: False).

        Returns:
            int: The next node in the random walk.
        """
        if previous_node is None:
            rand_index = np.random.choice(len(neighbors))
            next_node = neighbors[rand_index]
        else:
            normalized_probs = self.transition_probs[current_node][1]
            rand_index = np.random.choice(len(neighbors), p=normalized_probs)
            next_node = neighbors[rand_index]

        # Update the node embeddings using gradient descent
        target_embedding = self.node_embeddings[current_node]
        context_embedding = self.node_embeddings[next_node]

        if not negative:
            # Positive sample
            error = sigmoid(np.sum(target_embedding * context_embedding)) - 1
        else:
            # Negative sample
            error = sigmoid(np.sum(target_embedding * context_embedding))

        # Compute the gradient
        grad = error

        # Update the embeddings
        self.node_embeddings[current_node] -= self.learning_rate * grad * context_embedding
        self.node_embeddings[next_node] -= self.learning_rate * grad * target_embedding

        return next_node



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
            epoch_bar = CustomTQDM(total=self.num_walks * len(self.graph.nodes()), desc=f"Epoch {epoch + 1}/{self.epochs}",
                        leave=False, dynamic_ncols=True)

            with ProcessPoolExecutor() as executor:
                futures = []
                for _ in range(self.num_walks):
                    for node in self.graph.nodes():
                        futures.append(executor.submit(self.node2vec_walk, node))

                for future in futures:
                    walk = future.result()
                    loss = self.update_embeddings(walk, epoch_bar)
                    total_loss += loss

                epoch_bar.close()
                loss_history.append(total_loss / (self.num_walks * len(self.graph.nodes())))

        self.plot_loss(loss_history)

#     def update_embeddings(self, walk, epoch_bar):
#         """
#         Update node embeddings based on random walks.

#         Args:
#             walk (list): The random walk sequence.
#             epoch_bar (CustomTQDM): The progress bar for the current epoch.

#         Returns:
#             float: The total loss for the walk.
#         """
#         total_loss = 0.0

#         for i, target_node in enumerate(walk):
#             positive_samples = [walk[j] for j in
#                                 range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)) if
#                                 i != j]
#             for context_node in positive_samples:
#                 loss = self.skipgram_loss(target_node, context_node)
#                 total_loss += loss

#                 # Update the embeddings
#                 self.node2vec_step(target_node, context_node)

#             negative_samples = np.random.choice(len(self.graph.nodes()), size=self.negative_samples)
#             for negative_context_node in negative_samples:
#                 loss = self.skipgram_loss(target_node, negative_context_node, negative=True)
#                 total_loss += loss

#                 # Update the embeddings
#                 self.node2vec_step(target_node, negative_context_node, negative=True)

#         # Update the epoch's progress bar with the current step
#         epoch_bar.update(1)

#         return total_loss

#     def update_embeddings(self, walk, epoch_bar):
#         """
#         Update node embeddings based on random walks.

#         Args:
#             walk (list): The random walk sequence.
#             epoch_bar (CustomTQDM): The progress bar for the current epoch.

#         Returns:
#             float: The total loss for the walk.
#         """
#         total_loss = 0.0

#         for i, target_node in enumerate(walk):
#             positive_samples = [walk[j] for j in
#                                 range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)) if
#                                 i != j]
#             for context_node in positive_samples:
#                 loss = self.skipgram_loss(target_node, context_node)
#                 total_loss += loss

#                 # Update the embeddings
#                 self.node2vec_step(target_node, context_node)

#             negative_samples = np.random.choice(len(self.graph.nodes()), size=self.negative_samples)
#             for negative_context_node in negative_samples:
#                 loss = self.skipgram_loss(target_node, negative_context_node, negative=True)
#                 total_loss += loss

#                 # Update the embeddings
#                 self.node2vec_step(target_node, negative_context_node, negative=True)

#         # Update the epoch's progress bar with the current step
#         epoch_bar.update(1)

#         return total_loss

    def update_embeddings(self, walk, epoch_bar):
        """
        Update node embeddings based on random walks.

        Args:
            walk (list): The random walk sequence.
            epoch_bar (CustomTQDM): The progress bar for the current epoch.

        Returns:
            float: The total loss for the walk.
        """
        total_loss = 0.0

        for i, target_node in enumerate(walk):
            positive_samples = [walk[j] for j in
                                range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)) if
                                i != j]
            for context_node in positive_samples:
                loss = self.skipgram_loss(target_node, context_node)
                total_loss += loss

                # Update the embeddings
                self.node2vec_step(target_node, walk[i - 1] if i > 0 else None, positive_samples)

            negative_samples = np.random.choice(len(self.graph.nodes()), size=self.negative_samples)
            for negative_context_node in negative_samples:
                loss = self.skipgram_loss(target_node, negative_context_node, negative=True)
                total_loss += loss

                # Update the embeddings
                self.node2vec_step(target_node, None, negative_samples, negative=True)

        # Update the epoch's progress bar with the current step
        epoch_bar.update(1)

        return total_loss

    def skipgram_loss(self, target_node, context_node, negative=False):
        """
        Calculate the loss for skip-gram with negative sampling.

        Args:
            target_node: The embedding of the target node.
            context_node: The embedding of the context node (positive or negative).
            negative (bool, optional): Whether it's a negative sample (default: False).

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
