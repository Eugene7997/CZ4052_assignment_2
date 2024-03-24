# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: cz4052_assignment2
#     language: python
#     name: python3
# ---

# # Import statements

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
import time
# np.set_printoptions(threshold=sys.maxsize)

# # utility functions


def write_to_file(filename, data):
    with open(filename, "w") as file:
        for element in data:
            file.write(f"{element}\n")


def plot_the_scores(plot_values):
    """
    For plotting the utilities of the grid through the iteration of the agent's algorithm.
    """
    plt.plot(plot_values, label=f"Scores")

    plt.xlabel("Iterations")
    plt.ylabel("Utilities")
    plt.title("Utilities against Iterations for Each Position")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()


# # PageRank algorithms

def page_rank_simplified(adjacency_matrix, max_iterations=1000):
    # For plotting purposes
    plot_scores = []

    # Initialize the PageRank scores with equal probabilities (random surfers)
    number_of_nodes = adjacency_matrix.shape[0]
    page_rank_scores = np.ones(number_of_nodes) / number_of_nodes

    print(f"initial page_rank_scores:{page_rank_scores}")

    # Iteratively update the PageRank scores
    for _ in range(max_iterations):
        # Perform the matrix-vector multiplication
        new_page_rank_scores = adjacency_matrix.T.dot(page_rank_scores)

        # Check for convergence
        if np.allclose(page_rank_scores, new_page_rank_scores):
            print(f"convergence reached")
            break

        # For plotting
        plot_scores.append(page_rank_scores.round(4).tolist())

        page_rank_scores = new_page_rank_scores

        if np.isnan(new_page_rank_scores).any():
            print(f"iteration:{_}")

    return page_rank_scores, plot_scores


def page_rank_random_surfer_model(adjacency_matrix, teleportation_probability, max_iterations=100, threshold=0.0):
    plot_scores = []

    # Initialize the PageRank scores with equal probabilities (random surfers)
    number_of_nodes = adjacency_matrix.shape[0]
    page_rank_scores = np.ones(number_of_nodes) / number_of_nodes

    print(f"initial page_rank_scores:{page_rank_scores}")

    # Iteratively update the PageRank scores
    for _ in range(max_iterations):
        # Perform the matrix-vector multiplication
        new_page_rank_scores = adjacency_matrix.T.dot(page_rank_scores)
        # new_page_rank_scores = adjacency_matrix.dot(page_rank_scores)

        # Add the teleportation probability
        # new_page_rank_scores = (teleportation_probability * np.ones(number_of_nodes)) + ((1 - teleportation_probability) * new_page_rank_scores)
        new_page_rank_scores = (teleportation_probability * 1 / number_of_nodes) + ((1 - teleportation_probability) * new_page_rank_scores)
        # Check for convergence
        if np.linalg.norm(page_rank_scores - new_page_rank_scores) < threshold:
            break

        plot_scores.append(page_rank_scores.round(4).tolist())
        page_rank_scores = new_page_rank_scores

    return page_rank_scores, plot_scores


def modified_page_rank(adjacency_matrix, teleportation_probability, boredom_distribution, max_iterations=100):
    plot_scores = []

    # Initialize the PageRank scores with equal probabilities (random surfers)
    number_of_nodes = adjacency_matrix.shape[0]
    page_rank_scores = np.ones(number_of_nodes) / number_of_nodes

    print(f"initial page_rank_scores:{page_rank_scores}")

    # Iteratively update the PageRank scores
    for i in range(max_iterations):
        # Perform the matrix-vector multiplication
        new_page_rank_scores = adjacency_matrix.T.dot(page_rank_scores)

        # Add the teleportation probability
        new_page_rank_scores = (teleportation_probability * boredom_distribution) + ((1 - teleportation_probability) * new_page_rank_scores)
        # Check for convergence
        if np.allclose(page_rank_scores, new_page_rank_scores):
            break

        plot_scores.append(page_rank_scores.tolist())

        page_rank_scores = new_page_rank_scores

    return page_rank_scores, plot_scores, i


# # Proof of convergence with Example from lecture notes 7 page 53
# ![image.png](attachment:image.png)

# ## Graph set up

# +
# Transition matrix of above graph
a = np.array([[0, 1 / 2, 0, 0], [1 / 3, 0, 0, 1 / 2], [1 / 3, 0, 1, 1 / 2], [1 / 3, 1 / 2, 0, 0]])
a = np.transpose(a)

# G = nx.DiGraph(a)
G = nx.from_numpy_array(a, create_using=nx.DiGraph)

fig, ax = plt.subplots()
pos = nx.spring_layout(G, seed=5)
nx.draw_networkx_nodes(G, pos, ax=ax)
nx.draw_networkx_labels(G, pos, ax=ax)
nx.draw_networkx_edges(G, pos, ax=ax, connectionstyle=f"arc3, rad = {0.25}")
plt.title("Graph with Edge Labels")
plt.show()


# -

# ## Simplifed page rank algorithm


b, plot_scores = page_rank_simplified(a)
print(f"closed form 1: {b.round(2)}")
print(f"closed form 2: {b.sum().round(2)}")


write_to_file("./results/simplified_page_rank_scores_small_graph.txt", plot_scores)

plot_the_scores(plot_scores)


# # Parameter exploration of PageRank algorithm

# ## Experimentation with larger graphs

# datasets:
#
# https://langvillea.people.cofc.edu/PRDataCode/index.html?referrer=webcluster
#
# https://www.cs.cornell.edu/courses/cs685/2002fa/
#
# https://www.limfinity.com/ir/
#
# https://snap.stanford.edu/data/web-Google.html

# +
def parse_dataset(file_path):
    nodes = set()
    edges = []

    with open(file_path, "r") as file:
        for line in file:
            parts = line.split()
            if parts[0] == "n":
                nodes.add(int(parts[1]))
            elif parts[0] == "e":
                edges.append((int(parts[1]), int(parts[2])))

    return nodes, edges

# def create_adjacency_matrix(nodes, edges):
#     num_nodes = max(nodes) + 1
#     adjacency_matrix = np.zeros((num_nodes, num_nodes))

#     for edge in edges:
#         adjacency_matrix[edge[0]][edge[1]] = 1

#     for i in range(len(adjacency_matrix)):
#         row = adjacency_matrix[i]
#         row_sum = sum(row)

#         for j in range(len(row)):
#             # Avoid division by zero
#             if row_sum != 0:
#                 adjacency_matrix[i][j] = row[j] / row_sum
#             else:
#                 adjacency_matrix[i][j] = 1 / num_nodes

#     return adjacency_matrix, None

def create_adjacency_matrix_drop_dangling(nodes, edges):
    num_nodes = max(nodes) + 1
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    for edge in edges:
        adjacency_matrix[edge[0]][edge[1]] = 1

    # recursively drop dead nodes. dropping one node can cause another which linked only to it to become a dead end
    def drop_dead_nodes(matrix):
        dead_nodes = []
        for i in range(len(matrix)):
            row = matrix[i]
            if np.sum(row) == 0:
                dead_nodes.append(i)
        if len(dead_nodes) == 0:
            return matrix
        else:
            matrix = np.delete(matrix, dead_nodes, axis=0)
            matrix = np.delete(matrix, dead_nodes, axis=1)
            return drop_dead_nodes(matrix)

    adjacency_matrix = drop_dead_nodes(adjacency_matrix)

    for i in range(len(adjacency_matrix)):
        row = adjacency_matrix[i]
        row_sum = np.sum(row)
        if row_sum != 0:
            adjacency_matrix[i] = row / row_sum

    return adjacency_matrix, None

def create_adjacency_list(nodes, edges):
    adjacency_list = {node: [] for node in nodes}

    for edge in edges:
        source, destination = edge
        adjacency_list[source].append(destination)

    # handle dangling links but removing nodes with no outlinks
    adjacency_list_new = {node: outlinks for node, outlinks in adjacency_list.items() if outlinks}
    
    return adjacency_list_new


# -

# file_path = './datasets/gr0.epa_sample.txt'
# file_path = './datasets/gr0.epa_sample_dangling.txt'
# file_path = "./datasets/gr0.epa.txt"
# file_path = "./datasets/lecture_example.txt"
file_path = "./datasets/input.txt"
nodes, edges = parse_dataset(file_path)

adjacency_matrix, dropped_nodes = create_adjacency_matrix_drop_dangling(nodes, edges)
adjacency_list = create_adjacency_list(nodes, edges)

adjacency_matrix

# G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
# fig, ax = plt.subplots()
# pos = nx.spring_layout(G, seed=5)
# nx.draw_networkx_nodes(G, pos, ax=ax)
# nx.draw_networkx_labels(G, pos, ax=ax)
# nx.draw_networkx_edges(G, pos, ax=ax, connectionstyle=f"arc3, rad = {0.25}")
# plt.title("Graph with Edge Labels")
# plt.show()

# ## Web matrix exploration

def create_adjacency_matrix_zero(nodes, edges):
    num_nodes = max(nodes) + 1
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    for edge in edges:
        adjacency_matrix[edge[0]][edge[1]] = 1

    for i in range(len(adjacency_matrix)):
        row = adjacency_matrix[i]
        row_sum = sum(row)

        for j in range(len(row)):
            # Avoid division by zero
            if row_sum != 0:
                adjacency_matrix[i][j] = row[j] / row_sum
            else:
                # for nodes without outgoing edges, set the ranks to 0
                adjacency_matrix[i][j] = 0.0

    return adjacency_matrix


adjacency_matrix_zero = create_adjacency_matrix_zero(nodes, edges)
E = np.ones(adjacency_matrix_zero.shape[0]) / adjacency_matrix_zero.shape[0]
b, _, _ = modified_page_rank(adjacency_matrix_zero, 0.15, E)
# print(f"closed form 1: {b.round(2)}")
# print(f"closed form 2: {b.sum().round(2)}")
print(f"closed form 1: {b.round(2)}")
print(f"closed form 2: {b.sum().round(2)}")

E = np.ones(adjacency_matrix.shape[0]) / adjacency_matrix.shape[0]
b, _, _ = modified_page_rank(adjacency_matrix, 0.15, E)
print(f"closed form 1: {b.round(6)}")
print(f"closed form 2: {b.sum().round(6)}")

a = nx.pagerank(G, alpha=0.85)
print(f"closed form 1: {a}")
print(f"closed form 2: {sum(a.values())}")

index = 3
print(b[index])
print(a[index])

nx.draw(G, with_labels=True)

# ## Teleportation probability

# +
E = np.ones(adjacency_matrix.shape[0]) / adjacency_matrix.shape[0]
b, _, _ = modified_page_rank(adjacency_matrix, 0.15, E)
# print(f"closed form 1: {b.round(2)}")
# print(f"closed form 2: {b.sum().round(2)}")
print(f"closed form 1: {b}")
print(f"closed form 2: {b.sum()}")
print(f"max value: {np.max(b.round(6))}")
print(f"unique values: {np.unique(b.round(6))}")

c, _, _ = modified_page_rank(adjacency_matrix, 0.85, E)
print(f"closed form 1: {c}")
print(f"closed form 2: {c.sum()}")
print(f"max value: {np.max(c.round(6))}")
print(f"unique values: {np.unique(c.round(6))}")


filename = "./results/teleporation_probability_experiment.txt"
with open(filename, 'w') as f:
    print("=== 0.15 ===", file=f)
    print(f"closed form 1: {b}", file=f)
    print(f"closed form 2: {b.sum()}", file=f)
    print(f"max value: {np.max(b.round(6))}", file=f)
    print(f"unique values: {np.unique(b.round(6))}", file=f)
    print("\n")
    print("=== 0.85 ===", file=f)
    print(f"closed form 1: {c}", file=f)
    print(f"closed form 2: {c.sum()}", file=f)
    print(f"max value: {np.max(c.round(6))}", file=f)
    print(f"unique values: {np.unique(c.round(6))}", file=f)


# -

# ## Distribution vector E

def compute_boredom_distribution_linearly(num_nodes):
    E = np.linspace(0.0, 1.0, num_nodes)
    E = E / E.sum()
    return E


E = np.ones(adjacency_matrix.shape[0]) / adjacency_matrix.shape[0]
b, b_plot_scores, b_no_of_iterations = modified_page_rank(adjacency_matrix, 0.15, E)
# print(f"closed form 1: {b.round(2)}")
# print(f"closed form 2: {b.sum().round(2)}")
print(f"closed form 1: {b}")
print(f"closed form 2: {b.sum()}")
print(f"b_no_of_iterations: {b_no_of_iterations}")

E = compute_boredom_distribution_linearly(adjacency_matrix.shape[0])
c, c_plot_scores, c_no_of_iterations = modified_page_rank(adjacency_matrix, 0.15, E)
# print(f"closed form 1: {c.round(2)}")
# print(f"closed form 2: {c.sum().round(2)}")
print(f"closed form 1: {c}")
print(f"closed form 2: {c.sum()}")
print(f"c_no_of_iterations: {c_no_of_iterations}")

pr = nx.pagerank(G, 0.85)
array = np.array(list(pr.values()), dtype=float)
print(array)
print(array.sum())


# # PageRank with parallel programming

def modified_page_rank_speed_comparision(adjacency_matrix, teleportation_probability, boredom_distribution, max_iterations=100):
    # Initialize the PageRank scores with equal probabilities (random surfers)
    number_of_nodes = adjacency_matrix.shape[0]
    page_rank_scores = np.ones(number_of_nodes) / number_of_nodes

    print(f"initial page_rank_scores:{page_rank_scores}")

    # Iteratively update the PageRank scores
    for i in range(max_iterations):
        # Perform the matrix-vector multiplication
        new_page_rank_scores = adjacency_matrix.T.dot(page_rank_scores)

        # Add the teleportation probability
        new_page_rank_scores = (teleportation_probability * boredom_distribution) + ((1 - teleportation_probability) * new_page_rank_scores)
        # Check for convergence
        if np.allclose(page_rank_scores, new_page_rank_scores):
            break


        page_rank_scores = new_page_rank_scores

    return page_rank_scores, i

E = np.ones(adjacency_matrix.shape[0]) / adjacency_matrix.shape[0]
start_time = time.time()
b, _= modified_page_rank_speed_comparision(adjacency_matrix, 0.15, E)
time_taken = time.time() - start_time
print(f"closed form 1: {b}")
print(f"closed form 2: {b.sum()}")
print(f"time_taken:{time_taken}")


# ## Experimenting

# +
def pagerank_mapper(webpage, outlinks, pagerank_values):
    # Emit the webpage with an empty list if it's a dangling node
    if not outlinks:
        yield webpage, []
    for outlink in outlinks:
        yield outlink, pagerank_values[webpage] / len(outlinks)

def pagerank_reducer(webpage, values, damping_factor, total_webpages):
    outlink_list = []
    pagerank = 0
    for value in values:
        if isinstance(value, list):
            outlink_list.extend(value)
        else:
            pagerank += value
    
    # Handle dangling nodes by redistributing their PageRank evenly
    if not outlink_list:
        pagerank += damping_factor * pagerank / (total_webpages-1)
    
    new_pagerank = damping_factor * pagerank + (1 - damping_factor) / total_webpages
    return new_pagerank

def map_reduce(input_data, mapper, reducer, max_iterations=1000, convergence_threshold=1e-6, damping_factor=0.85):
    # pagerank_values = {webpage: 1.0 / len(input_data) for webpage in input_data}
    pagerank_values = {webpage: 1.0 for webpage in input_data}
    # new_pagerank_values = {key: 0.0 for key in input_data.keys()}
    # new_pagerank_values = {webpage: 1.0 for webpage in input_data}

    for iteration in range(max_iterations):
        intermediate_results = {}
        new_pagerank_values = {}
        
        for webpage, outlinks in input_data.items():
            mapped_pairs = mapper(webpage, outlinks, pagerank_values)
            for key, value in mapped_pairs:
                if key not in intermediate_results:
                    intermediate_results[key] = []
                intermediate_results[key].append(value)

        for key, values in intermediate_results.items():
            reduced_value = reducer(key, values, damping_factor, len(input_data))
            new_pagerank_values[key] = reduced_value

        # Handle dangling nodes
        for webpage in input_data.keys():
            if webpage not in new_pagerank_values:
                new_pagerank_values[webpage] = damping_factor / len(input_data)
    
        total_pagerank = sum(new_pagerank_values.values())
        new_pagerank_values = {webpage: pagerank / total_pagerank for webpage, pagerank in new_pagerank_values.items()}
        
        # convergence = sum(abs(new_pagerank_values[node] - pagerank_values[node]) for node in input_data.keys()) < convergence_threshold
        convergence = all(abs((new_pagerank_values[node] - pagerank_values[node]) / pagerank_values[node]) < convergence_threshold for node in input_data.keys())

        if convergence:
            print(f"convergence reached at iteration {iteration}")
            break

        pagerank_values = new_pagerank_values

    return pagerank_values

input_data = adjacency_list
damping_factor = 0.85

# Run MapReduce job
pagerank_results = map_reduce(input_data, pagerank_mapper, pagerank_reducer, convergence_threshold=0, damping_factor=damping_factor)
print(f"pagerank results:\n {pagerank_results}")
print(f"sum of page ranks: {sum(pagerank_results.values())}")
# -


b = nx.pagerank(G, 0.85)
print(f"pagerank results:\n {b}")
print(f"sum of page ranks: {sum(b.values())}")

index = 14
print(b[index])
print(pagerank_results[index])
