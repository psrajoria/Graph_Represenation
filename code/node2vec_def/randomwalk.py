import numpy as np
from gensim.models import Word2Vec
import networkx as nx

class GraphEmbedding:
    def __init__(self, graph, return_param, in_out_param, num_walks, walk_length, window_size, embedding_dimension, negative_samples):
        self.graph = graph
        self.return_param = return_param
        self.in_out_param = in_out_param
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.window_size = window_size
        self.embedding_dimension = embedding_dimension
        self.negative_samples = negative_samples
        self.transition_probs = None

    def calculate_transition_probabilities(self):
        G = self.graph
        transition_probs = {}

        for source_node in G.nodes():
            transition_probs[source_node] = {"neighbors": {}}
            for current_node in G.neighbors(source_node):
                probabilities = []

                for destination_node in G.neighbors(current_node):
                    if source_node == destination_node:
                        probability = G[current_node][destination_node].get("weight", 1) * (1 / self.return_param)
                    elif destination_node in G.neighbors(source_node):
                        probability = G[current_node][destination_node].get("weight", 1)
                    else:
                        probability = G[current_node][destination_node].get("weight", 1) * (1 / self.in_out_param)

                    probabilities.append(probability)

                transition_probs[source_node]["neighbors"][current_node] = probabilities / np.sum(probabilities)

        self.transition_probs = transition_probs
        return transition_probs

    def generate_random_walks(self):
        G = self.graph
        random_walks = []
        num_nodes = len(G.nodes())
        similarity_matrix = np.zeros((num_nodes, num_nodes))

        for start_node in G.nodes():
            for i in range(self.num_walks):
                walk = [start_node]
                walk_options = list(G[start_node])

                if not walk_options:
                    break

                for _ in range(self.walk_length - 1):
                    current_node = walk[-1]
                    probabilities = self.transition_probs[walk[-1]]["neighbors"][current_node]
                    next_step = np.random.choice(walk_options, p=probabilities)
                    walk.append(next_step)

                    # Update the similarity matrix
                    for node_j in G.nodes():
                        if node_j == start_node:
                            similarity_matrix[start_node][node_j] = 1.000  # Probability of visiting itself is 1
                        elif node_j not in walk:
                            similarity_matrix[start_node][node_j] = 0.000  # Probability of not visiting is 0
                        else:
                            # Calculate the probability of visiting node_j during the walk
                            prob_ij = walk.count(node_j) / (self.walk_length - 1)
                            similarity_matrix[start_node][node_j] += prob_ij

                random_walks.append(walk)

        np.random.shuffle(random_walks)
        random_walks = [list(map(str, walk)) for walk in random_walks]

        return random_walks, similarity_matrix

    def train_node_embeddings(self, walks, dimension=None, window=None, num_negative_samples=None):
        dimension = dimension or self.embedding_dimension
        window = window or self.window_size
        num_negative_samples = num_negative_samples or self.negative_samples

        model = Word2Vec(
            sentences=walks,
            window=window,
            vector_size=dimension,
            sg=1,
            negative=num_negative_samples,
        )
        return model.wv
