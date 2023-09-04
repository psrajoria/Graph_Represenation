# import scipy.io
# import networkx as nx
# import matplotlib.pyplot as plt
# import os


# class GraphReader:
#     def read_graph_from_txt(self, file_path):
#         try:
#             nodes = set()
#             edges = []

#             with open(file_path, "r") as file:
#                 lines = file.readlines()

#                 # Read the edges
#                 for line in lines:
#                     if not line.startswith("#"):
#                         from_node, to_node = map(int, line.strip().split())
#                         nodes.add(from_node)
#                         nodes.add(to_node)
#                         edges.append((from_node, to_node))

#             # Create a NetworkX graph
#             G = nx.Graph()
#             G.add_nodes_from(nodes)
#             G.add_edges_from(edges)

#             return nodes, edges, G

#         except Exception as e:
#             print("An error occurred:", e)
#             return None, None, None

#     def read_mat_file(self, mat_file):
#         try:
#             # Load data from the .mat file
#             data = scipy.io.loadmat(mat_file)

#             # Get the adjacency matrix from 'network'
#             adjacency_matrix = data["network"]

#             # Convert the adjacency matrix to a NetworkX graph
#             G = nx.from_scipy_sparse_array(adjacency_matrix)

#             # Extract nodes and edges from the graph
#             nodes = list(G.nodes())
#             edges = list(G.edges())

#             return nodes, edges, G

#         except Exception as e:
#             print("An error occurred:", e)
#             return None, None, None

#     def read_and_store_data(self, file_path):
#         file_extension = os.path.splitext(file_path)[1]

#         if file_extension == ".txt":
#             nodes, edges, graph = self.read_graph_from_txt(file_path)
#             return nodes, edges, graph
#         elif file_extension == ".mat":
#             nodes, edges, graph = self.read_mat_file(file_path)
#             return nodes, edges, graph
#         else:
#             print(f"Unsupported file format for {file_path}")
#             return None, None, None

#     def read_files_in_directory(self, directory_path):
#         files = os.listdir(directory_path)
#         data = {}

#         for file in files:
#             file_path = os.path.join(directory_path, file)
#             if os.path.isfile(file_path):
#                 nodes, edges, graph = self.read_and_store_data(file_path)
#                 if nodes is not None and edges is not None and graph is not None:
#                     data[file] = {"nodes": nodes, "edges": edges, "graph": graph}

#         return data

import scipy.io
import networkx as nx
import os


class GraphReader:
    ## Read Graph from TEXT file
    def read_graph_from_txt(self, file_path):
        try:
            nodes = set()
            edges = []

            with open(file_path, "r") as file:
                for line in file:
                    if not line.startswith("#"):
                        from_node, to_node = map(int, line.strip().split())
                        nodes.update((from_node, to_node))
                        edges.append((from_node, to_node))

            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)

            return nodes, edges, G

        except Exception as e:
            print("An error occurred:", e)
            return None, None, None

    ## Read Graph from MAT file
    def read_mat_file(self, mat_file):
        try:
            data = scipy.io.loadmat(mat_file)
            adjacency_matrix = data["network"]
            G = nx.from_scipy_sparse_array(adjacency_matrix)

            nodes = G.nodes()
            edges = G.edges()

            return nodes, edges, G

        except Exception as e:
            print("An error occurred:", e)
            return None, None, None

    ## Read Datasets
    def read_and_store_data(self, file_path):
        file_extension = os.path.splitext(file_path)[1]

        if file_extension == ".txt":
            return self.read_graph_from_txt(file_path)
        elif file_extension == ".mat":
            return self.read_mat_file(file_path)
        else:
            print(f"Unsupported file format for {file_path}")
            return None, None, None

    ## Read files in the directory
    def read_files_in_directory(self, directory_path):
        data = {}

        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                nodes, edges, graph = self.read_and_store_data(file_path)
                if nodes and edges and graph:
                    data[file] = {"nodes": nodes, "edges": edges, "graph": graph}

        return data
