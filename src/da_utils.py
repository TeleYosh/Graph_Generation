import os
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from extract_feats import extract_numbers
from utils import handle_nan
import networkx as nx
import community.community_louvain as community_louvain


"""
This file contains utility functions for exploratory data analysis (EDA) of the graph properties dataset.
At the end, there are functions to inspect the generated graphs.
"""


def get_train_and_valid_data_df(dataset, properties):
    """Load the graph properties dataset for training or validation set and return a DataFrame."""
    data = {}

    n = len(os.listdir(f"../data/{dataset}/description/"))

    for index in range(n):
        with open(f"../data/{dataset}/description/graph_{index}.txt", "r") as file:
            content = file.read()
            desc = content.split("\n")[0]
            stats = extract_numbers(desc)
            data[f"graph_{index}"] = stats

    return pd.DataFrame(data, index=properties).T


def get_test_data_df(properties):
    """Load the graph properties dataset for test set and return a DataFrame."""
    data = {}

    with open("../data/test/test.txt", "r") as file:
        content = file.read()
        graph_descriptions = content.split("\n")

    for graph_description in graph_descriptions:
        if not graph_description:
            continue
        stats = extract_numbers(graph_description)
        graph_id = stats[0]
        data[f"graph_{graph_id}"] = stats[1:]

    return pd.DataFrame(data, index=properties).T


def load_graph_properties():
    """Load the graph properties dataset for all sets."""
    properties = [
        "num_nodes",
        "num_edges",
        "avg_degree",
        "num_triangles",
        "global_clustering_coefficient",
        "k_core",
        "num_communities",
    ]

    df_train = get_train_and_valid_data_df("train", properties)
    df_valid = get_train_and_valid_data_df("valid", properties)
    df_test = get_test_data_df(properties)

    print("Number of training samples:", len(df_train))
    print("Number of validation samples:", len(df_valid))
    print("Number of test samples:", len(df_test))

    return df_train, df_valid, df_test


def global_statistics(df):
    """Compute global statistics of the dataset."""
    stats = df.describe().T
    # columns = ["min", "max", "mean", "std", "25%", "50%", "75%"]
    columns = ["min", "max", "mean", "std"]
    return stats[columns]


def perform_pca(df):
    """Return the principal components and explained variance of the properties dataset."""
    df = (df - df.mean()) / df.std()

    # Graph of the explained variance ratio
    pca_7 = PCA(n_components=7)
    pca_7.fit(df)
    print("Explained Variance Ratio:", pca_7.explained_variance_ratio_)
    plt.figure(figsize=(8, 6))
    plt.bar(
        np.arange(1, 8),
        pca_7.explained_variance_ratio_,
        alpha=0.7,
        label="Explained Variance Ratio",
    )
    plt.plot(
        np.arange(1, 8),
        pca_7.explained_variance_ratio_.cumsum(),
        marker="o",
        label="Cumulative Explained Variance Ratio",
    )
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Explained Variance Ratio of PCA")
    plt.legend()
    plt.show()

    # 2D PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df)
    explained_variance = pca.explained_variance_ratio_
    return principal_components, explained_variance


def plot_pca(principal_components, explained_variance):
    """Plot the 2D PCA of the properties dataset."""
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], s=5)
    plt.xlabel(f"Principal Component 1 ({explained_variance[0]*100:.2f}%)")
    plt.ylabel(f"Principal Component 2 ({explained_variance[1]*100:.2f}%)")
    plt.title("PCA of Graph Properties")
    plt.show()


def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Graph Properties")
    plt.show()


def plot_histogram_grid(df):
    properties = [
        "num_nodes",
        "num_edges",
        "avg_degree",
        "num_triangles",
        "global_clustering_coefficient",
        "k_core",
        "num_communities",
    ]

    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    axes = axes.flatten()

    for i, prop in enumerate(properties):
        sns.histplot(df[prop], bins=20, ax=axes[i])
        axes[i].set_title(f"Distribution of {prop}")
        axes[i].set_xlabel(prop)
        axes[i].set_ylabel("Frequency")

    # Remove the last two empty subplots
    for j in range(len(properties), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def parse_edge_list(edge_list):
    res = []
    for e in edge_list:
        if e == "":
            continue
        split = e.split(" ")
        res.append((int(split[0]), int(split[1])))
    return res


def get_graph(index, dataset="train"):
    assert dataset in ["train", "valid"], 'Dataset must be either "train" or "valid"'

    with open(f"../data/{dataset}/graph/graph_{index}.edgelist", "r") as file:
        content = file.read()
        edge_list = content.split("\n")

    parsed_edge_list = parse_edge_list(edge_list)
    G = nx.from_edgelist(parsed_edge_list)

    return G


def plot_random_graphs(n, dataset="train"):
    """
    Sample random graph indices, fetche the corresponding graphs and plot them.
    """
    n_train, n_valid = 8000, 1000
    assert dataset in ["train", "valid"], 'Dataset must be either "train" or "test".'
    assert (
        n <= n_train if dataset == "train" else n <= n_valid
    ), "n must be less than the number of graphs in the dataset."
    assert np.sqrt(n) % 1 == 0, "n must be a perfect square"

    indices = np.random.choice(
        range(n_train if dataset == "train" else n_valid), n, replace=False
    )
    plt.figure(figsize=(15, 15))

    for i, index in enumerate(indices):
        plt.subplot(int(np.sqrt(n)), int(np.sqrt(n)), i + 1)
        G = get_graph(index, dataset)
        nx.draw(
            G,
            with_labels=True,
            node_size=300,
            node_color="skyblue",
            font_size=10,
            font_weight="bold",
            edge_color="gray",
        )
        plt.title(f"Graph {index} ({dataset})")

    plt.show()


# To inspect the generated graphs
def calculate_stats_graph(G):
    # Number of nodes
    num_nodes = handle_nan(float(G.number_of_nodes()))
    # Number of edges
    num_edges = handle_nan(float(G.number_of_edges()))
    # Avg degree
    degrees = [deg for node, deg in G.degree()]
    avg_degree = handle_nan(float(sum(degrees) / len(degrees)))
    # Number of triangles
    triangles = nx.triangles(G)
    num_triangles = handle_nan(float(sum(triangles.values()) // 3))
    global_clustering_coefficient = handle_nan(float(nx.transitivity(G)))
    # Maximum k-core
    max_k_core = handle_nan(float(max(nx.core_number(G).values())))
    # calculate communities
    partition = community_louvain.best_partition(G)
    num_communities = handle_nan(float(len(set(partition.values()))))

    return [num_nodes, num_edges, avg_degree, num_triangles, global_clustering_coefficient, max_k_core, num_communities]