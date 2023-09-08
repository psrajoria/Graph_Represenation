import numpy as np
import torch
from gensim.models import Word2Vec
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import lil_matrix
import os
import datetime


class GraphEmbedding:
    def __init__(self, graph, return_param, in_out_param, num_walks, walk_length):
        self.graph = graph
        self.return_param = return_param
        self.in_out_param = in_out_param
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.transition_probs = None

    def calculate_transition_probabilities(self):
        G = self.graph
        transition_probs = {}
        return_param = self.return_param
        in_out_param = self.in_out_param

        for source_node in tqdm(G.nodes(), desc="Calculating Transition Probabilities"):
            transition_probs[source_node] = {"probabilities": {}}
            for current_node in G.neighbors(source_node):
                probabilities = list()

                for destination_node in G.neighbors(current_node):
                    if source_node == destination_node:
                        probability = G[current_node][destination_node].get("weight", 1) * (1 / return_param)
                    elif destination_node in G.neighbors(source_node):
                        probability = G[current_node][destination_node].get("weight", 1)
                    else:
                        probability = G[current_node][destination_node].get("weight", 1) * (1 / in_out_param)

                    probabilities.append(probability)

                sum_probabilities = np.sum(probabilities)
#                 transition_probs[source_node]["probabilities"][current_node] = [p / sum_probabilities for p in probabilities]
                transition_probs[source_node]["probabilities"][current_node] = probabilities / np.sum(probabilities)

        self.transition_probs = transition_probs
        return transition_probs

    def generate_random_walks(self):
        G = self.graph
        random_walks = list()
        device='cuda' if torch.cuda.is_available() else 'cpu'

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
                    probabilities = torch.tensor(probabilities, device='cuda' if torch.cuda.is_available() else 'cpu')

                    next_step = np.random.choice(walk_options, p=probabilities.cpu().numpy())
                    walk.append(next_step)

                random_walks.append(walk)
        unshuflled_random_walks = [list(map(str, walk)) for walk in random_walks]
        np.random.shuffle(random_walks)

        random_walks = [list(map(str, walk)) for walk in random_walks]
        return random_walks,unshuflled_random_walks


    def training(self, walks, window_size, dimension, epochs, batch_words=1000, sg=1, negative=5, learning_rate=0.025,min_count=5):
        model = Word2Vec(
            sentences=walks,
            window=window_size,
            vector_size=dimension,
            compute_loss=True,
            epochs=epochs,
            batch_words=batch_words,
            alpha=learning_rate,
            sg=sg,
            negative=negative,
            min_count=min_count,
            seed = 42
        )

        loss_values = []
        total_examples = len(walks)
        for epoch in tqdm(range(epochs), desc="Training Word2Vec"):
#             model.train(walks, total_examples=model.corpus_count, epochs=1)
            model.train(walks, total_examples=total_examples, epochs=1)
            loss = model.get_latest_training_loss()
            loss_values.append(loss)

        plt.plot(loss_values, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Word2Vec Training Loss')

        # Create a folder with the current date and time, and training parameters as the name
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        today_date = datetime.date.today().strftime("%d_%m_%Y")
        param_names = f"window{window_size}_dim{dimension}_epochs{epochs}_batch{batch_words}_sg{sg}_neg{negative}_lr{learning_rate}_mn{min_count}"
        
        folder_name = f"loss_plots_{today_date}"

        # Check if the folder exists, and if not, create it
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Save the plot in the folder with a filename including the timestamp
        plot_filename = os.path.join(folder_name, f"word2vec_training_loss_{param_names}_{timestamp}.png")
        plt.savefig(plot_filename)

        # Show the plot (optional)
        plt.show()

        return model.wv

    def calculate_probability_matrix(self, random_walks):
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
