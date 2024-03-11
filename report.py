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

# import sys
# np.set_printoptions(threshold=sys.maxsize)

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


def page_rank_simplified(adjacency_matrix, max_iterations=1000):
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

        page_rank_scores = new_page_rank_scores

        if np.isnan(new_page_rank_scores).any():
            print(f"iteration:{_}")

    return page_rank_scores


b = page_rank_simplified(a)
print(f"closed form 1: {b.round(2)}")
print(f"closed form 2: {b.sum().round(2)}")


# # Parameter exploration of PageRank algorithm

# ## Addition of teleportation probability


def page_rank_random_surfer_model(adjacency_matrix, teleportation_probability, max_iterations=100):
    # Initialize the PageRank scores with equal probabilities (random surfers)
    number_of_nodes = adjacency_matrix.shape[0]
    page_rank_scores = np.ones(number_of_nodes) / number_of_nodes

    print(f"initial page_rank_scores:{page_rank_scores}")

    # Iteratively update the PageRank scores
    for _ in range(max_iterations):
        # Perform the matrix-vector multiplication
        new_page_rank_scores = adjacency_matrix.T.dot(page_rank_scores)

        # Add the teleportation probability
        # new_page_rank_scores = (teleportation_probability * np.ones(number_of_nodes)) + ((1 - teleportation_probability) * new_page_rank_scores)
        new_page_rank_scores = (teleportation_probability * 1 / number_of_nodes) + ((1 - teleportation_probability) * new_page_rank_scores)
        # Check for convergence
        if np.allclose(page_rank_scores, new_page_rank_scores):
            break

        page_rank_scores = new_page_rank_scores

    return page_rank_scores


b = page_rank_random_surfer_model(a, 0.2)
# should be 0.101351351	0.128378378	0.641891892	0.128378378
print(b.round(2))
print(b.sum().round(2))


# ## Modified pagerank model


def modified_page_rank(adjacency_matrix, teleportation_probability, boredom_distribution, max_iterations=100):
    # Initialize the PageRank scores with equal probabilities (random surfers)
    number_of_nodes = adjacency_matrix.shape[0]
    page_rank_scores = np.ones(number_of_nodes) / number_of_nodes

    print(f"initial page_rank_scores:{page_rank_scores}")

    # Iteratively update the PageRank scores
    for _ in range(max_iterations):
        # Perform the matrix-vector multiplication
        new_page_rank_scores = adjacency_matrix.T.dot(page_rank_scores)

        # Add the teleportation probability
        new_page_rank_scores = (teleportation_probability * boredom_distribution) + ((1 - teleportation_probability) * new_page_rank_scores)
        # Check for convergence
        if np.allclose(page_rank_scores, new_page_rank_scores):
            break

        page_rank_scores = new_page_rank_scores
        return page_rank_scores


# boredom_distribution = np.array([0.25, 0.25, 0.25, 0.25])
boredom_distribution = np.array([0.74, 0.24, 0.01, 0.01])
b = modified_page_rank(a, 0.15, boredom_distribution)
print(b)
print(b.sum())

# ## Experimenting with teleportation probability

b = page_rank_random_surfer_model(a, 0.01)
print(b.round(2))
print(b.sum().round(2))

b = page_rank_random_surfer_model(a, 0.99)
print(b.round(2))
print(b.sum().round(2))

# Higher Teleportation probability increases uniform distribution

# ## Experimenting with distribution vector

boredom_distribution = np.array([0.7, 0.1, 0.1, 0.1])
b = modified_page_rank(a, 0.15, boredom_distribution)
print(b.round(2))
print(b.sum().round(2))

boredom_distribution = np.array([0.25, 0.25, 0.25, 0.25])
b = modified_page_rank(a, 0.20, boredom_distribution)
print(b.round(2))
print(b.sum().round(2))

# Even with an even distribution, the addition of distribution vector, E has made pagerank scores much fairer. e.g [0.15 0.22 0.42 0.22] vs [0.1  0.13 0.64 0.13].

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
import numpy as np


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
            # Avoid division by zero
            if row_sum != 0:
                adjacency_matrix[i][j] = row[j] / row_sum
            else:
                adjacency_matrix[i][j] = 0.0

    return adjacency_matrix


# -

file_path = "./datasets/gr0.epa_sample.txt"  # Replace with the actual path to your dataset
nodes, edges = parse_dataset(file_path)
adjacency_matrix = create_adjacency_matrix(nodes, edges)

# +
G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

fig, ax = plt.subplots()
pos = nx.spring_layout(G, seed=5)
nx.draw_networkx_nodes(G, pos, ax=ax)
nx.draw_networkx_labels(G, pos, ax=ax)
nx.draw_networkx_edges(G, pos, ax=ax, connectionstyle=f"arc3, rad = {0.25}")
plt.title("Graph with Edge Labels")
plt.show()
# -

b = page_rank_random_surfer_model(adjacency_matrix, 0.15)
print(f"closed form 1: {b.round(2)}")
print(f"closed form 2: {b.sum().round(2)}")

# # PageRank with parallel programming
