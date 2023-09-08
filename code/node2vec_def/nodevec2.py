import random
import numpy as np
import networkx as nx

class Node2Vec:
    def __init__(self, graph, p, q):
        """
        Initialize Node2Vec with graph, p, and q parameters.

        Parameters:
        - graph (dict): The graph represented as an adjacency list.
        - p (float): Parameter p for the Node2Vec algorithm.
        - q (float): Parameter q for the Node2Vec algorithm.
        """
        self.graph = graph
        self.p = p
        self.q = q
        self.edge_weights = self.calculate_edge_weights()

    def calculate_edge_weights(self):
        """
        Calculate edge weights for each node in the graph.

        Returns:
        - edge_weights (dict): Edge weights for each node.
        """
        edge_weights = {}
        for node, neighbors in self.graph.items():
            total_weight = sum(neighbors.values())
            edge_weights[node] = {neighbor: weight / total_weight for neighbor, weight in neighbors.items()}
        return edge_weights

    def node2vec_random_walk(self, start_node, walk_length):
        """
        Generate a Node2Vec random walk starting from a given node.

        Parameters:
        - start_node (str): The starting node for the random walk.
        - walk_length (int): Length of the random walk.

        Returns:
        - walk (list): The generated random walk as a list of nodes.
        """
        walk = [start_node]

        for _ in range(walk_length - 1):
            current_node = walk[-1]
            neighbors = self.graph[current_node]
            if not neighbors:
                break

            # Calculate unnormalized transition probabilities using edge weights and parameters p and q
            probabilities = []
            for neighbor in neighbors:
                if neighbor == walk[-2]:  # Exclude the previous node
                    probabilities.append(1.0 / self.p)
                else:
                    dtx = self.shortest_path_distance(walk[-2], neighbor)
                    edge_weight = self.graph[current_node][neighbor]
                    delta_tx = self.calculate_delta(dtx)
                    probability = delta_tx * edge_weight
                    probabilities.append(probability)

            # Choose the next node based on the probabilities
            next_node = np.random.choice(neighbors, p=probabilities)
            walk.append(next_node)

        return walk

    def calculate_delta(self, dtx):
        """
        Calculate delta function based on shortest path distance.

        Parameters:
        - dtx (int): Shortest path distance between nodes t and x.

        Returns:
        - delta (float): Delta value based on distance and parameters p and q.
        """
        if dtx == 0:
            return 1.0 / self.p
        elif dtx == 1:
            return 1.0
        elif dtx == 2:
            return 1.0 / self.q

    def shortest_path_distance(self, node1, node2):
        """
        Calculate the shortest path distance between two nodes using NetworkX.

        Parameters:
        - node1 (str): First node.
        - node2 (str): Second node.

        Returns:
        - distance (int): Shortest path distance between the nodes.
        """
        if node1 == node2:
            return 0  # Distance to the same node is 0
        try:
            distance = nx.shortest_path_length(self.graph, source=node1, target=node2)
            return distance
        except nx.NetworkXNoPath:
            return float('inf')  # Return infinity for nodes not connected

    def generate_random_walks(self, num_walks, walk_length):
        """
        Generate random walks for the entire graph.

        Parameters:
        - num_walks (int): Number of random walks to generate per node.
        - walk_length (int): Length of each random walk.

        Returns:
        - walks (list of lists): List of generated random walks.
        """
        walks = []
        nodes = list(self.graph.keys())

        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self.node2vec_random_walk(node, walk_length)
                walks.append(walk)

        return walks