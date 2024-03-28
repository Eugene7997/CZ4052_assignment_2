# ---
# jupyter:
#   jupytext:
#     formats: py,ipynb
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
import os
import random
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

# ## Simplified pagerank

def page_rank_simplified(adjacency_matrix, max_iterations=1000):
    # For plotting purposes
    plot_scores = []

    # Initialize the PageRank scores with equal probabilities (random surfers)
    number_of_nodes = adjacency_matrix.shape[0]
    page_rank_scores = np.ones(number_of_nodes) / number_of_nodes

    print(f"initial page_rank_scores:{page_rank_scores}")

    # Iteratively update the PageRank scores
    for i in range(max_iterations):
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
            print(f"iteration:{i}")

    return page_rank_scores, plot_scores, i


# ## Random surfer model pagerank

def page_rank_random_surfer_model(adjacency_matrix, teleportation_probability, max_iterations=100, threshold=0.0):
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
        new_page_rank_scores = (teleportation_probability * 1 / number_of_nodes) + ((1 - teleportation_probability) * new_page_rank_scores)
        # Check for convergence
        if np.linalg.norm(page_rank_scores - new_page_rank_scores) < threshold:
            break

        plot_scores.append(page_rank_scores.round(4).tolist())
        page_rank_scores = new_page_rank_scores

    return page_rank_scores, plot_scores, i


# ## modified pagerank with teleportation and distribution vector

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

# ## Graph set up

# +
# Transition matrix of above graph
lecture_notes_example_adjacency_matrix = np.array([[0, 1 / 2, 0, 0], [1 / 3, 0, 0, 1 / 2], [1 / 3, 0, 1, 1 / 2], [1 / 3, 1 / 2, 0, 0]])
lecture_notes_example_adjacency_matrix = np.transpose(lecture_notes_example_adjacency_matrix)

G = nx.from_numpy_array(lecture_notes_example_adjacency_matrix, create_using=nx.DiGraph)

fig, ax = plt.subplots()
pos = nx.spring_layout(G, seed=5)
nx.draw_networkx_nodes(G, pos, ax=ax)
nx.draw_networkx_labels(G, pos, ax=ax)
nx.draw_networkx_edges(G, pos, ax=ax, connectionstyle=f"arc3, rad = {0.25}")
plt.title("Graph with Edge Labels")
plt.show()
# -

# ## Proofs

page_ranks, simplified_pagerank_lecture_notes_example_plot_scores, no_of_iteration = page_rank_simplified(lecture_notes_example_adjacency_matrix)
print(f"closed form 1: {page_ranks.round(2)}")
print(f"closed form 2: {page_ranks.sum().round(2)}")
print(f"no_of_iteration: {no_of_iteration}")

write_to_file("./results/simplified_page_rank_scores_lecture_notes_example.txt", simplified_pagerank_lecture_notes_example_plot_scores)
plot_the_scores(simplified_pagerank_lecture_notes_example_plot_scores)

page_ranks, random_surfer_model_pagerank_lecture_notes_example_plot_scores, no_of_iteration = page_rank_random_surfer_model(lecture_notes_example_adjacency_matrix, 0.15)
print(f"closed form 1: {page_ranks.round(2)}")
print(f"closed form 2: {page_ranks.sum().round(2)}")
print(f"no_of_iteration: {no_of_iteration}")


write_to_file("./results/random_surfer_page_rank_scores_lecture_notes_example.txt", random_surfer_model_pagerank_lecture_notes_example_plot_scores)
plot_the_scores(random_surfer_model_pagerank_lecture_notes_example_plot_scores)


E = np.ones(lecture_notes_example_adjacency_matrix.shape[0]) / lecture_notes_example_adjacency_matrix.shape[0]
page_ranks, modified_pagerank_lecture_notes_example_plot_scores, no_of_iteration = modified_page_rank(lecture_notes_example_adjacency_matrix, 0.15, E)
print(f"closed form 1: {page_ranks.round(2)}")
print(f"closed form 2: {page_ranks.sum().round(2)}")
print(f"no_of_iteration: {no_of_iteration}")

write_to_file("./results/modified_page_rank_scores_lecture_notes_example.txt", modified_pagerank_lecture_notes_example_plot_scores)
plot_the_scores(modified_pagerank_lecture_notes_example_plot_scores)


# # Parameter exploration of PageRank algorithm

# ## Experiment set up

# +
def generate_input_data(num_nodes, num_edges, file_name):
    nodes = [f"n {i}" for i in range(num_nodes)]
    edges = [f"e {random.randint(0, num_nodes-1)} {random.randint(0, num_nodes-1)}" for _ in range(num_edges)]
    directed_edges = [f"e {random.randint(0, num_nodes-1)} {random.randint(0, num_nodes-1)}" for _ in range(num_edges)]
    input_data =  nodes + directed_edges
    with open(f"{file_name}.txt", "w") as f:
        for line in input_data:
            f.write(f"{line}\n")

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

def create_adjacency_matrix(nodes, edges):
    num_nodes = max(nodes) + 1
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    for edge in edges:
        adjacency_matrix[edge[0]][edge[1]] = 1

    for i in range(len(adjacency_matrix)):
        row = adjacency_matrix[i]
        row_sum = sum(row)

        for j in range(len(row)):
            if row_sum != 0:
                adjacency_matrix[i][j] = row[j] / row_sum

    return adjacency_matrix

def create_adjacency_matrix_replace_dead_nodes(nodes, edges):
    num_nodes = max(nodes) + 1
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    for edge in edges:
        adjacency_matrix[edge[0]][edge[1]] = 1

    for i in range(len(adjacency_matrix)):
        row = adjacency_matrix[i]
        row_sum = sum(row)

        for j in range(len(row)):
            if row_sum != 0:
                adjacency_matrix[i][j] = row[j] / row_sum
            else:
                adjacency_matrix[i][j] = 1 / num_nodes

    return adjacency_matrix

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

def create_adjacency_list_replace(nodes, edges):
    adjacency_list = {node: [] for node in nodes}

    for edge in edges:
        source, destination = edge
        adjacency_list[source].append(destination)

    # handle dangling links by replacing them with links to all nodes
    num_nodes = len(adjacency_list)
    dead_nodes = [node for node in nodes if not adjacency_list[node]]
    if dead_nodes:
        for dead_node in dead_nodes:
            adjacency_list[dead_node] = list(adjacency_list.keys())
            adjacency_list[dead_node].remove(dead_node)
    print(f"dead_nodes:{dead_nodes}")
    return adjacency_list

def create_adjacency_list_remove(nodes, edges):
    adjacency_list = {node: [] for node in nodes}

    for edge in edges:
        source, destination = edge
        adjacency_list[source].append(destination)

    # handle dangling links by removing nodes with no outlinks
    adjacency_list_new = {node: outlinks for node, outlinks in adjacency_list.items() if outlinks}
    
    return adjacency_list_new


# -

# ## Experimentation with larger graphs

# +
# desired_num_nodes = 20
# desired_num_edges = 40
# # Ensure num_edges is less than or equal to num_nodes^2 to avoid creating more edges than possible in a directed graph.
# num_edges = min(desired_num_edges, desired_num_nodes**2)
# generate_input_data(desired_num_nodes, desired_num_edges, "medium_graph")
# -

# # Dataset source: https://www.cs.cornell.edu/courses/cs685/2002fa/data/gr0.epa
# file_path = './datasets/gr0.epa_sample.txt'
# file_path = './datasets/gr0.epa_sample_dangling.txt'
# file_path = './datasets/gr0.epa_sample_dangling_2.txt'
# file_path = "./datasets/gr0.epa.txt"
# file_path = "./datasets/lecture_example.txt"
file_path = "./datasets/medium_graph.txt"
nodes, edges = parse_dataset(file_path)

# graph creation
G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

nx.draw(G, with_labels=True)


# ## Web matrix

# ### normal adjacency matrix

adjacency_matrix = create_adjacency_matrix(nodes, edges)

E = np.ones(adjacency_matrix.shape[0]) / adjacency_matrix.shape[0]
page_ranks, normal_web_matrix_simplified_pagerank_plot_scores, no_of_iterations = page_rank_simplified(adjacency_matrix)
print(f"closed form 1: {page_ranks}")
print(f"closed form 2: {page_ranks.sum()}") # NOT converged to 1!
print(f"no_of_iterations: {no_of_iterations}")

# ### remove dead nodes recursively in web matrix

adjacency_matrix, dropped_nodes = create_adjacency_matrix_drop_dangling(nodes, edges)

page_ranks, remove_deadnodes_web_matrix_simplified_pagerank_plot_scores, no_of_iterations = page_rank_simplified(adjacency_matrix)
print(f"closed form 1: {page_ranks}")
print(f"closed form 2: {page_ranks.sum()}")
print(f"no_of_iterations: {no_of_iterations}")

# ### replace deadnodes with special nodes

adjacency_matrix = create_adjacency_matrix_replace_dead_nodes(nodes, edges)

E = np.ones(adjacency_matrix.shape[0]) / adjacency_matrix.shape[0]
page_ranks, replace_deadnodes_web_matrix_simplified_pagerank_plot_scores, no_of_iterations = page_rank_simplified(adjacency_matrix)
print(f"closed form 1: {page_ranks}")
print(f"closed form 2: {page_ranks.sum()}")
print(f"no_of_iterations: {no_of_iterations}")

z = nx.pagerank(G, 0.85)
print(f"pagerank results:\n {z}")
print(f"sum of page ranks: {sum(z.values())}")

# ## Teleportation probability

adjacency_matrix = create_adjacency_matrix_replace_dead_nodes(nodes, edges)

# +
pagerank_15_percent, pagerank_15_percent_plot_scores, pagerank_15_percent_no_of_iteration = page_rank_random_surfer_model(adjacency_matrix, 0.15)
print(f"closed form 1: {pagerank_15_percent}")
print(f"closed form 2: {pagerank_15_percent.sum()}")
print(f"max value: {np.max(pagerank_15_percent.round(6))}")
print(f"unique values: {np.unique(pagerank_15_percent.round(6))}")

pagerank_85_percent, pagerank_85_percent_plot_scores, pagerank_85_percent_no_of_iteration = page_rank_random_surfer_model(adjacency_matrix, 0.85)
print(f"closed form 1: {pagerank_85_percent}")
print(f"closed form 2: {pagerank_85_percent.sum()}")
print(f"max value: {np.max(pagerank_85_percent.round(6))}")
print(f"unique values: {np.unique(pagerank_85_percent.round(6))}")


filename = "./results/teleporation_probability_experiment.txt"
with open(filename, 'w') as f:
    print("=== 0.15 ===", file=f)
    print(f"closed form 1: {pagerank_15_percent}", file=f)
    print(f"closed form 2: {pagerank_15_percent.sum()}", file=f)
    print(f"max value: {np.max(pagerank_15_percent.round(6))}", file=f)
    print(f"mean value: {np.mean(pagerank_15_percent)}", file=f)
    print(f"standard deviation: {np.std(pagerank_15_percent)}", file=f)
    print(f"no_of_iterations: {pagerank_15_percent_no_of_iteration}", file=f)
    print("\n")
    print("=== 0.85 ===", file=f)
    print(f"closed form 1: {pagerank_85_percent}", file=f)
    print(f"closed form 2: {pagerank_85_percent.sum()}", file=f)
    print(f"max value: {np.max(pagerank_85_percent.round(6))}", file=f)
    print(f"mean value: {np.mean(pagerank_85_percent)}", file=f)
    print(f"standard deviation: {np.std(pagerank_85_percent)}", file=f)
    print(f"no_of_iterations: {pagerank_85_percent_no_of_iteration}", file=f)
# -

plot_the_scores(pagerank_15_percent_plot_scores)

plot_the_scores(pagerank_85_percent_plot_scores)


# ## Distribution vector E

def compute_boredom_distribution_linearly(num_nodes):
    E = np.linspace(0.0, 1.0, num_nodes)
    E = E / E.sum()
    return E


# +
E = np.ones(adjacency_matrix.shape[0]) / adjacency_matrix.shape[0]
pagerank_average_e, pagerank_average_e_plot_scores, pagerank_average_e_no_of_iteration = modified_page_rank(adjacency_matrix, 0.15, E)
print(f"closed form 1: {pagerank_average_e}")
print(f"closed form 2: {pagerank_average_e.sum()}")
print(f"b_no_of_iterations: {pagerank_average_e_no_of_iteration}")

E = compute_boredom_distribution_linearly(adjacency_matrix.shape[0])
pagerank_linear_e, pagerank_linear_e_plot_scores, pagerank_linear_e_no_of_iteration = modified_page_rank(adjacency_matrix, 0.15, E)
print(f"closed form 1: {pagerank_linear_e}")
print(f"closed form 2: {pagerank_linear_e.sum()}")
print(f"c_no_of_iterations: {pagerank_linear_e_no_of_iteration}")


filename = "./results/distribution_vector_experiment.txt"
with open(filename, 'w') as f:
    print("=== average ===", file=f)
    print(f"closed form 1: {pagerank_average_e}", file=f)
    print(f"closed form 2: {pagerank_average_e.sum()}", file=f)
    print(f"max value: {np.max(pagerank_average_e.round(6))}", file=f)
    print(f"mean value: {np.mean(pagerank_average_e)}", file=f)
    print(f"standard deviation: {np.std(pagerank_average_e)}", file=f)
    print(f"no_of_iterations: {pagerank_average_e_no_of_iteration}", file=f)
    print("\n")
    print("=== linear ===", file=f)
    print(f"closed form 1: {pagerank_linear_e}", file=f)
    print(f"closed form 2: {pagerank_linear_e.sum()}", file=f)
    print(f"max value: {np.max(pagerank_linear_e.round(6))}", file=f)
    print(f"mean value: {np.mean(pagerank_linear_e)}", file=f)
    print(f"standard deviation: {np.std(pagerank_linear_e)}", file=f)
    print(f"no_of_iterations: {pagerank_linear_e_no_of_iteration}", file=f)
# -

# # Map reduce

adjacency_list = create_adjacency_list_replace(nodes, edges)

adjacency_list


# +
def mapper(damping_factor, nodes, pagerank, new_pagerank):
    for node, neighbors in nodes.items():
        for neighbor in neighbors:
            contribution = damping_factor * pagerank[node] / len(neighbors)
            new_pagerank[neighbor] += contribution
    return new_pagerank

def reducer(damping_factor, nodes, pagerank, new_pagerank, N):
    for node in nodes:
        pagerank[node] = new_pagerank.get(node, 0) + (1 - damping_factor) / N
    return pagerank

def calculate_pagerank(nodes, damping_factor=0.85, num_iterations=10):
    N = len(nodes)
    initial_pr = 1 / N
    pagerank = {node: initial_pr for node in nodes}

    for _ in range(num_iterations):
        new_pagerank = {node: 0 for node in nodes}
        # mapper 
        new_pagerank = mapper(damping_factor, nodes, pagerank, new_pagerank)
        # reducer
        pagerank = reducer(damping_factor, nodes, pagerank, new_pagerank, N)

    return pagerank

start_time = time.time()
pagerank_values = calculate_pagerank(adjacency_list)
time_taken = time.time() - start_time

# print the results
for node, pr in pagerank_values.items():
    print(f"Node {node}: PageRank = {pr:.4f}")
summation = sum(pagerank_values.values())
print(f"Summation: {summation}")
print(f"time_taken: {time_taken}")
# -

E = np.ones(adjacency_matrix.shape[0]) / adjacency_matrix.shape[0]
page_ranks, b_plot_scores, no_of_iterations = modified_page_rank(adjacency_matrix, 0.15, E)
print(f"closed form 1: {page_ranks}")
print(f"closed form 2: {page_ranks.sum()}")
print(f"no_of_iterations: {no_of_iterations}")
