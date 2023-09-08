import numpy as np
from gensim.models import Word2Vec
import networkx as nx


class GraphEmbedding:
    """
    A class for computing node embeddings from a graph using random walks and Word2Vec.

    Args:
        graph (networkx.Graph): The input graph.
        p (float): Return parameter p for the node2vec algorithm.
        q (float): In-out parameter q for the node2vec algorithm.
        num_walks (int): Number of random walks to perform per node.
        walk_len (int): Length of each random walk.
        window_size (int): The window size for the Word2Vec model.
        dim (int): The dimensionality of the node embeddings.
        negative (int): Number of negative samples to use for negative sampling.

    Attributes:
        graph (networkx.Graph): The input graph.
        p (float): Return parameter p for the node2vec algorithm.
        q (float): In-out parameter q for the node2vec algorithm.
        num_walks (int): Number of random walks to perform per node.
        walk_len (int): Length of each random walk.
        window_size (int): The window size for the Word2Vec model.
        dim (int): The dimensionality of the node embeddings.
        negative (int): Number of negative samples to use for negative sampling.
        probs (dict): Store transition probabilities.
    """

    def __init__(self, graph, params):
        self.graph = graph
        self.p = params["p"]
        self.q = params["q"]
        self.num_walks = params["num_walks"]
        self.walk_len = params["walk_len"]
        self.window_size = params["window_size"]
        self.dim = params["dim"]
        self.negative = params["negative"]
        self.probs = None  # Store transition probabilities

    def transition_probabilities(self):
        """
        Compute transition probabilities for random walks on the graph.

        Returns:
            dict: A dictionary containing the transition probabilities for each node.
        """
        G = self.graph
        probs = {}

        for source_node in G.nodes():
            probs[source_node] = {"probabilities": {}}
            for current_node in G.neighbors(source_node):
                probs_ = list()
                for destination in G.neighbors(current_node):
                    if source_node == destination:
                        prob_ = G[current_node][destination].get("weight", 1) * (
                            1 / self.p
                        )
                    elif destination in G.neighbors(source_node):
                        prob_ = G[current_node][destination].get("weight", 1)
                    else:
                        prob_ = G[current_node][destination].get("weight", 1) * (
                            1 / self.q
                        )

                    probs_.append(prob_)
                probs[source_node]["probabilities"][current_node] = probs_ / np.sum(
                    probs_
                )

        self.probs = probs  # Store transition probabilities
        return probs

    def random_walks(self):
        """
        Generate random walks on the graph.

        Returns:
            list: A list of random walks as sequences of node IDs.
        """
        G = self.graph
        walks = list()
        num_nodes = len(G.nodes())
        S = np.zeros((num_nodes, num_nodes))  # Initialize the similarity matrix

        for start_node in G.nodes():
            for i in range(self.num_walks):
                walk = [start_node]
                walk_options = list(G[start_node])

                if len(walk_options) == 0:
                    break

                first_step = np.random.choice(walk_options)
                walk.append(first_step)

                for k in range(self.walk_len - 2):
                    walk_options = list(G[walk[-1]])

                    if len(walk_options) == 0:
                        break

                    probabilities = self.probs[walk[-2]]["probabilities"][walk[-1]]
                    next_step = np.random.choice(walk_options, p=probabilities)
                    walk.append(next_step)

                walks.append(walk)
                # Update the similarity matrix
                for node_j in G.nodes():
                    if node_j == start_node:
                        S[start_node][
                            node_j
                        ] = 1.000  # Probability of visiting itself is 1
                    elif node_j not in walk:
                        S[start_node][
                            node_j
                        ] = 0.000  # Probability of not visiting is 0
                    else:
                        # Calculate the probability of visiting node_j during the walk
                        prob_ij = walk.count(node_j) / (self.walk_len - 1)
                        S[start_node][node_j] += prob_ij
        np.random.shuffle(walks)
        walks = [list(map(str, walk)) for walk in walks]

        return walks, S

    def train_embedding(
        self,
        walks,
        dim=None,
        window_size=None,
        negative=None,
        workers=None,
        epochs=None,
        batch_size=None,
        learning_rate=None,
        seed=None,
    ):
        """
        Train node embeddings using Word2Vec.

        Args:
            walks (list): List of random walks.
            dim (int, optional): The dimensionality of the node embeddings. If None, use the class's dim.
            window_size (int, optional): The window size for the Word2Vec model. If None, use the class's window_size.
            negative (int, optional): Number of negative samples to use for negative sampling.
                If None, use the class's negative value.

        Returns:
            gensim.models.keyedvectors.KeyedVectors: Trained Word2Vec model.
        """
        if dim is None:
            dim = self.dim
        if window_size is None:
            window_size = self.window_size
        if negative is None:
            negative = self.negative

        if epochs is None:
            epochs = 1  # Default to 1 epoch if not specified

        if batch_size is None:
            batch_size = 32  # Default batch size if not specified

        if learning_rate is None:
            learning_rate = 0.025  # Default learning rate if not specified

        loss_history = []  # Added to store loss values during training

        model = Word2Vec(
            sentences=walks,
            window=window_size,
            vector_size=dim,
            sg=1,
            negative=negative,
            workers=workers if workers is not None else 4,
            epochs=epochs,
            compute_loss=True,
            alpha=learning_rate,
            seed=seed,
        )

        # Extract the loss values from the model
        loss_values = model.get_latest_training_loss()
        loss_history.extend(loss_values)

        # Plot loss values if matplotlib is available
        if plt:
            plt.plot(loss_history)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Word2Vec Training Loss")
            plt.show()

        return model.wv
