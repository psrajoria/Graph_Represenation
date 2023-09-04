# import numpy as np
# import random
# import matplotlib.pyplot as plt
# import networkx as nx


# class Node2Vec:
#     def __init__(
#         self,
#         graph,
#         dimensions=64,
#         num_walks=10,
#         walk_length=80,
#         window_size=10,
#         num_epochs=100,
#         workers=4,
#         p=1.0,  # Return parameter
#         q=1.0,  # In-out parameter
#         learning_rate=0.025,
#     ):
#         self.graph = graph
#         self.dimensions = dimensions
#         self.num_walks = num_walks
#         self.walk_length = walk_length
#         self.window_size = window_size
#         self.num_epochs = num_epochs
#         self.workers = workers
#         self.p = p
#         self.q = q
#         self.learning_rate = learning_rate

#         # Generate random walks (training pairs are implicitly created)
#         self.walks = self.generate_walks()

#         # Initialize node embeddings
#         self.node_embeddings = {
#             node: np.random.rand(dimensions) for node in graph.nodes()
#         }

#         # Train embeddings
#         self.train_embedding()

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
#             # Calculate transition probabilities based on p and q
#             if neighbor == prev_node:
#                 weights.append(1.0 / self.p)
#             elif self.graph.has_edge(curr_node, neighbor):
#                 weights.append(1.0)
#             else:
#                 weights.append(1.0 / self.q)
#         weights = np.array(weights)
#         weights /= weights.sum()

#         # Choose the next node based on the computed probabilities
#         next_node = np.random.choice(neighbors, p=weights)
#         return next_node

#     def train_embedding(self):
#         # SGD optimization
#         for epoch in range(self.num_epochs):
#             random.shuffle(self.walks)
#             for walk in self.walks:
#                 for i, target_node in enumerate(walk):
#                     for context_node in walk[
#                         max(0, i - self.window_size) : i + self.window_size + 1
#                     ]:
#                         if target_node != context_node:
#                             target_embedding = self.node_embeddings[target_node]
#                             context_embedding = self.node_embeddings[context_node]
#                             error = self.skipgram_loss(
#                                 target_embedding, context_embedding
#                             )
#                             self.update_embeddings(
#                                 target_embedding, context_embedding, error
#                             )

#     def skipgram_loss(self, target_embedding, context_embedding):
#         dot_product = np.dot(target_embedding, context_embedding)
#         sigmoid = 1 / (1 + np.exp(-dot_product))
#         return -np.log(sigmoid)

#     def update_embeddings(self, target_embedding, context_embedding, error):
#         gradient = (1 - 1 / (1 + np.exp(-error))) * self.learning_rate
#         target_embedding -= gradient * context_embedding
#         context_embedding -= gradient * target_embedding

#     def plot_loss(self, save_path="skipgram_loss.png"):
#         # Initialize an empty list to store losses
#         losses = []

#         # Calculate and append the loss for each walk in the dataset
#         for walk in self.walks:
#             for i, target_node in enumerate(walk):
#                 for context_node in walk[
#                     max(0, i - self.window_size) : i + self.window_size + 1
#                 ]:
#                     if target_node != context_node:
#                         target_embedding = self.node_embeddings[target_node]
#                         context_embedding = self.node_embeddings[context_node]
#                         error = self.skipgram_loss(target_embedding, context_embedding)
#                         losses.append(error)

#         # Plot the losses
#         plt.figure(figsize=(10, 6))
#         plt.hist(losses, bins=50, alpha=0.7, color="b")
#         plt.title("Skipgram Loss Distribution")
#         plt.xlabel("Loss")
#         plt.ylabel("Frequency")

#         # Display the plot
#         plt.show()

#         # Save the plot as an image
#         plt.savefig(save_path, bbox_inches="tight")
#         print(f"Loss plot saved as {save_path}")

#     def embed_all_nodes(self):
#         return self.node_embeddings


# import numpy as np
# import random
# import matplotlib.pyplot as plt
# import networkx as nx
# import os
# import time  # Import the time module

# # Set a seed for reproducibility
# np.random.seed(42)
# random.seed(42)


# class Node2Vec:
#     def __init__(
#         self,
#         graph,
#         dimensions=64,
#         num_walks=10,
#         walk_length=80,
#         window_size=10,
#         num_epochs=100,
#         workers=4,
#         p=1.0,  # Return parameter
#         q=1.0,  # In-out parameter
#         learning_rate=0.025,
#         learning_rate_decay=1e-4,  # Add a learning rate decay
#     ):
#         self.graph = graph
#         self.dimensions = dimensions
#         self.num_walks = num_walks
#         self.walk_length = walk_length
#         self.window_size = window_size
#         self.num_epochs = num_epochs
#         self.workers = workers
#         self.p = p
#         self.q = q
#         self.learning_rate = learning_rate
#         self.learning_rate_decay = learning_rate_decay  # Added learning rate decay

#         # Generate random walks (training pairs are implicitly created)
#         self.walks = self.generate_walks()

#         # Initialize node embeddings with small random values centered around zero
#         self.node_embeddings = {
#             node: (np.random.rand(dimensions) - 0.5) * 0.01 for node in graph.nodes()
#         }

#         # Initialize a list to store losses for each epoch
#         self.losses_per_epoch = []

#         # Get the current date and time for the directory name
#         current_datetime = time.strftime("%Y%m%d-%H%M%S")

#         # Create a directory for saving loss plots with date and time
#         self.loss_dir = os.path.join(
#             "loss",
#             f"Loss_{current_datetime}_dim_{dimensions}_walks_{num_walks}_length_{walk_length}_window_{window_size}_p_{p}_q_{q}",
#         )
#         os.makedirs(self.loss_dir, exist_ok=True)

#         # Train embeddings
#         self.train_embedding()

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
#             # Calculate transition probabilities based on p and q
#             if neighbor == prev_node:
#                 weights.append(1.0 / self.p)
#             elif self.graph.has_edge(curr_node, neighbor):
#                 weights.append(1.0)
#             else:
#                 weights.append(1.0 / self.q)
#         weights = np.array(weights)
#         weights /= weights.sum()

#         # Choose the next node based on the computed probabilities
#         next_node = np.random.choice(neighbors, p=weights)
#         return next_node

#     def train_embedding(self):
#         # SGD optimization
#         for epoch in range(self.num_epochs):
#             random.shuffle(self.walks)
#             epoch_loss = 0.0
#             for walk in self.walks:
#                 for i, target_node in enumerate(walk):
#                     for context_node in walk[
#                         max(0, i - self.window_size) : i + self.window_size + 1
#                     ]:
#                         if target_node != context_node:
#                             target_embedding = self.node_embeddings[target_node]
#                             context_embedding = self.node_embeddings[context_node]
#                             error = self.skipgram_loss(
#                                 target_embedding, context_embedding
#                             )
#                             self.update_embeddings(
#                                 target_embedding, context_embedding, error
#                             )
#                             epoch_loss += error

#             # Apply learning rate decay
#             self.learning_rate *= 1.0 / (1.0 + self.learning_rate_decay * epoch)

#             # Append the epoch loss to the list
#             self.losses_per_epoch.append(epoch_loss)

#             # Print the loss for the current epoch
#             print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss}")

#         # After all epochs are completed, display and save the loss plot
#         self.plot_loss()

#     def skipgram_loss(self, target_embedding, context_embedding):
#         dot_product = np.dot(target_embedding, context_embedding)
#         log_sigmoid = -np.log(1 / (1 + np.exp(-dot_product)))
#         return log_sigmoid

#     def update_embeddings(self, target_embedding, context_embedding, error):
#         gradient = (1 - 1 / (1 + np.exp(-error))) * self.learning_rate
#         target_embedding -= gradient * context_embedding
#         context_embedding -= gradient * target_embedding

#     def plot_loss(self):
#         # Plot the losses for all epochs on the same graph as a line plot
#         plt.figure(figsize=(10, 6))
#         plt.plot(
#             range(1, self.num_epochs + 1),
#             self.losses_per_epoch,
#             marker="o",
#             linestyle="-",
#             color="b",
#         )
#         plt.title("Skipgram Loss Over Epochs")
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.grid(True)

#         # Save the plot in the loss directory with a filename
#         save_path = os.path.join(self.loss_dir, "skipgram_loss.png")
#         plt.savefig(save_path, bbox_inches="tight")
#         print(f"Loss plot saved as {save_path}")

#     def embed_all_nodes(self):
#         return self.node_embeddings


import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import os
import time  # Import the time module
from gensim.models import Word2Vec

# Set a seed for reproducibility
np.random.seed(42)
random.seed(42)


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
        learning_rate_decay=1e-4,  # Add a learning rate decay
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
        self.learning_rate_decay = learning_rate_decay  # Added learning rate decay

        # Generate random walks (training pairs are implicitly created)
        self.walks = self.generate_walks()

        # Initialize node embeddings with small random values centered around zero
        self.node_embeddings = {
            node: (np.random.rand(dimensions) - 0.5) * 0.01 for node in graph.nodes()
        }

        # Initialize a list to store losses for each epoch
        self.losses_per_epoch = []

        # Get the current date and time for the directory name
        current_datetime = time.strftime("%Y%m%d-%H%M%S")

        # Create a directory for saving loss plots with date and time
        self.loss_dir = os.path.join(
            "loss",
            f"Loss_{current_datetime}_dim_{dimensions}_walks_{num_walks}_length_{walk_length}_window_{window_size}_p_{p}_q_{q}",
        )
        os.makedirs(self.loss_dir, exist_ok=True)

        # Train embeddings
        self.train_embedding()

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
        # SGD optimization
        for epoch in range(self.num_epochs):
            random.shuffle(self.walks)
            epoch_loss = 0.0
            for walk in self.walks:
                for i, target_node in enumerate(walk):
                    for context_node in walk[
                        max(0, i - self.window_size) : i + self.window_size + 1
                    ]:
                        if target_node != context_node:
                            target_embedding = self.node_embeddings[target_node]
                            context_embedding = self.node_embeddings[context_node]
                            error = self.skipgram_loss(
                                target_embedding, context_embedding
                            )
                            self.update_embeddings(
                                target_embedding, context_embedding, error
                            )
                            epoch_loss += error

            # Apply learning rate decay
            self.learning_rate *= 1.0 / (1.0 + self.learning_rate_decay * epoch)

            # Append the epoch loss to the list
            self.losses_per_epoch.append(epoch_loss)

            # Print the loss for the current epoch
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss}")

        # After all epochs are completed, display and save the loss plot
        self.plot_loss()

        # Train Word2Vec embeddings
        self.word2vec_embeddings = self.train_word2vec()

    def skipgram_loss(self, target_embedding, context_embedding):
        dot_product = np.dot(target_embedding, context_embedding)
        log_sigmoid = -np.log(1 / (1 + np.exp(-dot_product)))
        return log_sigmoid

    def update_embeddings(self, target_embedding, context_embedding, error):
        gradient = (1 - 1 / (1 + np.exp(-error))) * self.learning_rate
        target_embedding -= gradient * context_embedding
        context_embedding -= gradient * target_embedding

    def plot_loss(self):
        # Plot the losses for all epochs on the same graph as a line plot
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, self.num_epochs + 1),
            self.losses_per_epoch,
            marker="o",
            linestyle="-",
            color="b",
            label="Node2Vec Loss",
        )

        # Plot Word2Vec loss (initially set to 0)
        plt.axhline(y=0, color="r", linestyle="--", label="Word2Vec Loss")

        plt.title("Loss Comparison Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # Save the plot in the loss directory with a filename
        save_path = os.path.join(self.loss_dir, "loss_comparison.png")
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Loss comparison plot saved as {save_path}")

    def train_word2vec(self):
        # Prepare data for Word2Vec training
        sentences = [list(map(str, walk)) for walk in self.walks]

        # Train Word2Vec model
        model = Word2Vec(
            sentences,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=1,
            sg=1,  # Skip-gram model
            workers=self.workers,
            epochs=self.num_epochs,
        )

        # Get Word2Vec embeddings for all nodes
        word2vec_embeddings = {
            str(node): model.wv[str(node)] for node in self.graph.nodes()
        }

        return word2vec_embeddings

    def embed_all_nodes(self):
        return self.node_embeddings, self.word2vec_embeddings
