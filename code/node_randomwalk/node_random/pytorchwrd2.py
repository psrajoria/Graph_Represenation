import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import datetime
from scipy.sparse import lil_matrix

# Import PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim

# Define the SkipGram model
class SkipGram(nn.Module):
    """
    SkipGram model for word embeddings.

    Args:
        vocab_size (int): Vocabulary size, the number of unique words.
        embedding_dim (int): Dimension of word embeddings.
        weight_decay (float, optional): Weight decay for the optimizer (L2 regularization). Default is 1e-5.

    Attributes:
        in_embed (nn.Embedding): Input embedding layer.
        out_embed (nn.Embedding): Output embedding layer.
        weight_decay (float): Weight decay for regularization.

    Methods:
        forward(target, context): Forward pass of the model.
    """

    def __init__(self, vocab_size, embedding_dim, weight_decay=1e-5):
        super(SkipGram, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)
        self.weight_decay = weight_decay

    def forward(self, target, context):
        """
        Forward pass of the SkipGram model.

        Args:
            target (torch.Tensor): Target word indices.
            context (torch.Tensor): Context word indices.

        Returns:
            scores (torch.Tensor): Word similarity scores.
        """
        target_embeds = self.in_embed(target)
        context_embeds = self.out_embed(context)
        scores = target_embeds @ context_embeds.t()
        return scores


class GraphEmbedding:
    """
    GraphEmbedding class for node embeddings using random walks on a graph.

    Args:
        graph (nx.Graph): Input graph.
        return_param (float): Return parameter for random walks.
        in_out_param (float): In-Out parameter for random walks.
        num_walks (int): Number of random walks per node.
        walk_length (int): Length of each random walk.

    Attributes:
        graph (nx.Graph): Input graph.
        return_param (float): Return parameter for random walks.
        in_out_param (float): In-Out parameter for random walks.
        num_walks (int): Number of random walks per node.
        walk_length (int): Length of each random walk.
        transition_probs (dict): Transition probabilities for nodes.

    Methods:
        calculate_transition_probabilities(): Calculate transition probabilities for random walks.
        generate_random_walks(): Generate random walks on the graph.
        training(walks, window_size, dimension, epochs, weight_decay=1e-5, learning_rate=0.025): Train word embeddings using SkipGram.
        calculate_probability_matrix(random_walks): Calculate a probability matrix based on common node visits in walks.
    """

    def __init__(self, graph, return_param, in_out_param, num_walks, walk_length):
        self.graph = graph
        self.return_param = return_param
        self.in_out_param = in_out_param
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.transition_probs = None

    def calculate_transition_probabilities(self):
        """
        Calculate transition probabilities for random walks.

        Returns:
            transition_probs (dict): Transition probabilities for nodes.
        """
        G = self.graph
        transition_probs = {}
        return_param = self.return_param
        in_out_param = self.in_out_param

        for source_node in tqdm(G.nodes(), desc="Calculating Transition Probabilities"):
            transition_probs[source_node] = {"probabilities": {}}
            for current_node in G.neighbors(source_node):
                probabilities = []

                for destination_node in G.neighbors(current_node):
                    if source_node == destination_node:
                        weight = G[current_node][destination_node].get("weight")
                        probability = weight / return_param if weight is not None else 1.0 / return_param
                    elif destination_node in G.neighbors(source_node):
                        weight = G[current_node][destination_node].get("weight")
                        probability = weight if weight is not None else 1.0
                    else:
                        weight = G[current_node][destination_node].get("weight")
                        probability = weight / in_out_param if weight is not None else 1.0 / in_out_param

                    probabilities.append(probability)

                transition_probs[source_node]["probabilities"][current_node] = probabilities / np.sum(probabilities)

        self.transition_probs = transition_probs
        return transition_probs

    def generate_random_walks(self):
        """
        Generate random walks on the graph.

        Returns:
            random_walks (list): List of random walks.
            unshuffled_random_walks (list): List of unshuffled random walks.
        """
        G = self.graph
        random_walks = list()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for start_node in tqdm(G.nodes(), desc="Generating Random Walks"):
            for i in range(self.num_walks):
                walk = [start_node]
                walk_options = list(G[start_node])

                if len(walk_options) == 0:
                    break
                first_step = np.random.choice(walk_options)
                walk.append(first_step)

                for _ in range(self.walk_length - 2):
                    walk_options = list(G[walk[-1]])
                    if len(walk_options) == 0:
                        break
                    current_node = walk[-1]
                    probabilities = self.transition_probs[walk[-2]]["probabilities"][current_node]

                    # Move probabilities to the GPU if available
                    probabilities = torch.tensor(probabilities, device=device)

                    next_step = np.random.choice(walk_options, p=probabilities.cpu().numpy())
                    walk.append(next_step)

                random_walks.append(walk)
        unshuffled_random_walks = [list(map(str, walk)) for walk in random_walks]
        np.random.shuffle(random_walks)

        random_walks = [list(map(str, walk)) for walk in random_walks]
        return random_walks, unshuffled_random_walks

    def training(self, walks, window_size, dimension, epochs, weight_decay=1e-5, learning_rate=0.025):
        """
        Train word embeddings using SkipGram.

        Args:
            walks (list): List of random walks.
            window_size (int): Window size for SkipGram.
            dimension (int): Dimension of word embeddings.
            epochs (int): Number of training epochs.
            weight_decay (float, optional): Weight decay for the optimizer (L2 regularization). Default is 1e-5.
            learning_rate (float, optional): Learning rate for optimization. Default is 0.025.

        Returns:
            model (SkipGram): Trained SkipGram model.
        """
        # Build vocabulary
        vocab = set(word for walk in walks for word in walk)
        word_to_index = {word: i for i, word in enumerate(vocab)}
        index_to_word = {i: word for word, i in word_to_index.items()}
        vocab_size = len(vocab)

        # Create the SkipGram model
        model = SkipGram(vocab_size, dimension, weight_decay)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        loss_values = []

        for epoch in tqdm(range(epochs), desc="Training Word2Vec"):
            total_loss = 0
            for walk in walks:
                for target_pos in range(len(walk)):
                    target_word = word_to_index[walk[target_pos]]
                    context_words = [
                        word_to_index[walk[i]]
                        for i in range(
                            max(0, target_pos - window_size),
                            min(len(walk), target_pos + window_size + 1)
                        )
                        if i != target_pos  # Exclude the target word itself
                    ]

                    target = torch.tensor([target_word], dtype=torch.long)
                    context = torch.tensor(context_words, dtype=torch.long)

                    scores = model(target, context)
                    loss = -torch.log(torch.sigmoid(scores)).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

            loss_values.append(total_loss)

        plt.plot(loss_values, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Word2Vec Training Loss')

        # Create a folder with the current date and time, and training parameters as the name
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        today_date = datetime.date.today().strftime("%d_%m_%Y")
        param_names = f"window{window_size}_dim{dimension}_epochs{epochs}_lr{learning_rate}"
        folder_name = f"loss_plots_{today_date}"

        # Check if the folder exists, and if not, create it
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Save the plot in the folder with a filename including the timestamp
        plot_filename = os.path.join(folder_name, f"word2vec_training_loss_{param_names}_{timestamp}.png")
        plt.savefig(plot_filename)

        # Show the plot (optional)
        plt.show()

        return model

    def calculate_probability_matrix(self, random_walks):
        """
        Calculate a probability matrix based on common node visits in walks.

        Args:
            random_walks (list): List of random walks.

        Returns:
            prob_matrix (np.ndarray): Probability matrix.
        """
        num_nodes = len(random_walks) // self.num_walks
        prob_matrix = np.zeros((num_nodes, num_nodes))

        for i in tqdm(range(num_nodes), desc="Calculating Probabilities"):
            for j in range(num_nodes):
                if i == j:
                    prob_matrix[i, j] = 1
                else:
                    common_count = 0

                    for k in range(self.num_walks):
                        walk_i = random_walks[i * self.num_walks + k]
                        walk_j = random_walks[j * self.num_walks + k]

                        visited_nodes_i = set()
                        visited_nodes_j = set()

                        for step in range(self.walk_length):
                            visited_nodes_i.add(walk_i[step])
                            visited_nodes_j.add(walk_j[step])

                        common_nodes = visited_nodes_i.intersection(visited_nodes_j)

                        if str(j) in common_nodes:
                            common_count += 1

                    prob_matrix[i, j] = common_count / self.num_walks

        return prob_matrix