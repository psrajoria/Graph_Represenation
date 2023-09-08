import os
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
import torch
from torch import Tensor
import random
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
import networkx as nxa
from networkx.algorithms import community
import numpy as np
from gensim.models import Word2Vec
import networkx as nx
import random
import networkx as nx
import random
import numpy as np
from typing import List
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec

import matplotlib.pyplot as plt
from node_random.randomwalk import GraphEmbedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "Data/"
os.makedirs(data_dir, exist_ok=True)
SEED = 42
dataset = Planetoid(root=data_dir, name="Cora")
data = dataset[0]

G = to_networkx(data, to_undirected=True)




def loss_function(Z, S):
    """
    Calculate the objective function ||Z^T*Z - S||^2.

    Args:
        Z (numpy.ndarray): The node embeddings as a numpy array.
        S (numpy.ndarray): The similarity matrix.

    Returns:
        float: The value of the objective function.
    """
    Z_transpose = np.transpose(Z)
    diff = np.dot(Z_transpose, Z) - S
    obj_func = np.linalg.norm(diff)
    return obj_func

# Define the hyperparameter search space
param_space = {
    'return_param': np.random.uniform(0.0, 4.0, 50),  # Example: random uniform sampling
    'in_out_param': np.random.uniform(0.0, 4.0, 50),  # Example: random uniform sampling
    'num_walks': np.random.randint(5, 15, 8),  # Example: random integer sampling
    'walk_length': np.random.randint(60, 100,10),  # Example: random integer sampling
    'window_size': np.random.randint(5, 10, 5),  # Example: random integer sampling
    'dimension': np.random.choice([32,64,128,256,512], 6),  # Example: random choice
    'epochs': np.random.choice([25, 50,75,100,125], 5),  # Example: random choice
    'negative': np.random.choice([3, 6], 5),  # Example: random choice
    'batch_words': np.random.choice([256, 512], 5),  # Example: random choice
    'learning_rate': np.random.choice([0.01, 0.0001, 0.02],10),
    'min_count': np.random.choice([0,1,2], 6),
}

best_loss = float('inf')
best_params = None

# Perform random search
num_iterations = 20  # Adjust the number of iterations as needed

for _ in range(num_iterations):
    # Randomly sample hyperparameters
    sampled_params = {param: np.random.choice(values) for param, values in param_space.items()}
    
    # Create an instance of your GraphEmbedding class with sampled hyperparameters
    embedding = GraphEmbedding(
        graph=G,
        return_param=sampled_params['return_param'],
        in_out_param=sampled_params['in_out_param'],
        num_walks=sampled_params['num_walks'],
        walk_length=sampled_params['walk_length'],
    )
    
    probab = embedding.calculate_transition_probabilities()
    # Train and evaluate with the current set of hyperparameters
    walks, uns = embedding.generate_random_walks()
    S = embedding.calculate_probability_matrix(uns)
    node_embeddings = embedding.training(
        walks,
        window_size=sampled_params['window_size'],
        dimension=sampled_params['dimension'],
        sg=1,
        epochs=sampled_params['epochs'],
        negative=sampled_params['negative'],
        batch_words=sampled_params['batch_words'],
        learning_rate = sampled_params['learning_rate']
    )
    Z = node_embeddings.vectors.T
    current_loss = loss_function(Z, S)
    
    # Update the best hyperparameters if the current set is better
    if current_loss < best_loss:
        best_loss = current_loss
        best_params = sampled_params
        
        
print("Best Hyperparameters:", best_params)
print("Best Loss:", best_loss)