__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2024, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

import json
import os
import sys
from itertools import combinations

import numpy as np
from matplotlib.patches import Rectangle
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import csr_matrix
import markov_clustering as mc
import networkx as nx
import matplotlib
import matplotlib.pyplot
from sklearn.metrics import silhouette_score, adjusted_rand_score, recall_score, precision_score, accuracy_score, \
    f1_score, cluster

from LaSSI.tests.benchmark import Benchmark
metrics_benchmark = Benchmark("Metrics")


def graph_plot(matrix, clusters, filename="graph.png"):
    fig = matplotlib.pyplot.figure()
    ## Color blind palette: https://github.com/mpetroff/accessible-color-cycles
    palette = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581",
               "#92dadd"]
    assert len(clusters) <= len(palette)
    graph = nx.Graph(matrix)
    cluster_map = {node: i for i, cluster in enumerate(clusters) for node in cluster}
    edges, weights = zip(*nx.get_edge_attributes(graph, 'weight').items())
    colors = [palette[cluster_map[i]] for i in range(len(graph.nodes()))]
    positions = nx.spring_layout(graph, seed=31)  ## ensuring the same layout for the same graph!
    nx.draw_networkx(
        graph, node_color=colors, pos=positions, with_labels=True, edgelist=edges, edge_color=weights,
        edge_cmap=matplotlib.cm.YlGnBu, ax=fig.add_subplot(), font_size=24, node_size=750, width=2
    )
    if filename is not None:
        # Save plot to file
        matplotlib.use("Agg")
        fig.savefig(filename, dpi=200, bbox_inches='tight')
    else:
        # Display interactive viewer
        matplotlib.pyplot.show()


def _plot_dendrogram(model, **kwargs):
    # Authors: Mathew Kallada
    # License: BSD 3 clause
    """
    =========================================
    Plot Hierarachical Clustering Dendrogram
    =========================================

    This example plots the corresponding dendrogram of a hierarchical clustering
    using AgglomerativeClustering and the dendrogram method available in scipy.
    https://github.com/scikit-learn/scikit-learn/blob/70cf4a676caa2d2dad2e3f6e4478d64bcb0506f7/examples/cluster/plot_hierarchical_clustering_dendrogram.py
    """

    from scipy.cluster.hierarchy import dendrogram
    # Children of hierarchical clustering
    children = model.children_
    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])
    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0] + 2)
    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, **kwargs)


def plot_dendogram(model, D, filename="dendrogram.png", box_clusters=None):
    # fig = matplotlib.pyplot.figure()
    from scipy.spatial.distance import squareform
    # Compute and plot first dendrogram.
    # condensedD = squareform(D)
    import scipy.cluster.hierarchy as sch
    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6])
    # Y = sch.linkage(condensedD, method='centroid')
    Z1 = _plot_dendrogram(model, orientation='left')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticklabels([], minor=False, fontsize=20)
    ax1.set_yticklabels([], minor=False, fontsize=20)

    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
    # Y = sch.linkage(condensedD, method='single')
    Z2 = _plot_dendrogram(model)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticklabels([], minor=False, fontsize=20)
    ax2.set_yticklabels([], minor=False, fontsize=20)

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    # assert idx1 == idx2
    # idx1 = list(idx1)
    # D = D[:, idx1][idx1]
    D = D[idx1, :]
    D = D[:, idx2]
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=matplotlib.cm.YlGnBu)
    # axmatrix.set_xticks([])  # remove axis labels
    # axmatrix.set_yticks([])  # remove axis labels
    #
    # # Plot colorbar.
    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.6])
    cbar = matplotlib.pyplot.colorbar(im, cax=axcolor)
    cbar.ax.tick_params(labelsize=20)
    axmatrix.set_xticks(range(len(idx1)))
    axmatrix.set_xticklabels(idx1, minor=False, fontsize=20)
    # axmatrix.xaxis.set_label_position('bottom')
    # axmatrix.xaxis.tick_bottom()
    #
    # pylab.xticks(rotation=-90, fontsize=8)
    #
    axmatrix.set_yticks(range(len(idx2)))
    axmatrix.set_yticklabels(idx2, minor=False, fontsize=20)
    # axmatrix.yaxis.set_label_position('right')
    # axmatrix.yaxis.tick_right()

    if box_clusters is not None:
        indices_x = [list(idx1).index(cluster) for cluster in box_clusters]
        indices_y = [list(idx2).index(cluster) for cluster in box_clusters]

        box_linewidth = 4
        box_color = 'red'

        if len(box_clusters) == 3:
            if is_consecutive(indices_x) and is_consecutive(indices_y):
                # Highlight the 3x3 grid
                x_start = min(indices_x) - 0.5
                y_start = max(indices_y) - 2.5
                width = 3
                height = 3
                rect = Rectangle((x_start, y_start), width, height, linewidth=box_linewidth, edgecolor=box_color,
                                 facecolor='none')
                axmatrix.add_patch(rect)
            else:
                # Highlight pairwise combinations
                for cluster_pair in combinations(box_clusters, 2):
                    cluster_label_x, cluster_label_y = cluster_pair
                    index_x = list(idx1).index(cluster_label_x)
                    index_y = list(idx2).index(cluster_label_y)

                    # Check for diagonal adjacency
                    highlight_boxes(axmatrix, box_color, box_linewidth, index_x, index_y)
        elif len(box_clusters) == 2:
            cluster_label_x, cluster_label_y = box_clusters
            index_x = list(idx1).index(cluster_label_x)
            index_y = list(idx2).index(cluster_label_y)
            n_y = len(idx2)  # Total number of labels on the y-axis

            # Check for diagonal adjacency
            highlight_boxes(axmatrix, box_color, box_linewidth, index_x, index_y)

    # axcolor = fig.add_axes([0.94, 0.1, 0.02, 0.6])
    # plt.show()
    # _plot_dendrogram(model, labels=model.labels_, ax=fig.add_subplot())
    if filename is not None:
        # Save plot to file
        matplotlib.use("Agg")
        fig.savefig(filename, dpi=200, bbox_inches='tight')
        matplotlib.pyplot.close()
    else:
        # Display interactive viewer
        matplotlib.pyplot.show()


def highlight_boxes(axmatrix, box_color, box_linewidth, index_x, index_y):
    if abs(index_x - index_y) == 1:
        # Highlight the 2x2 grid
        x_start = min(index_y, index_x) - 0.5
        y_start = max(index_x, index_y) - 1.5
        width = 2
        height = 2
        rect = Rectangle((x_start, y_start), width, height, linewidth=box_linewidth,
                         edgecolor=box_color,
                         facecolor='none')
        axmatrix.add_patch(rect)
    else:
        # Highlight individual boxes
        rect1 = Rectangle((index_y - 0.5, index_x - 0.5), 1, 1, linewidth=box_linewidth,
                          edgecolor=box_color,
                          facecolor='none')
        axmatrix.add_patch(rect1)
        rect2 = Rectangle((index_x - 0.5, index_y - 0.5), 1, 1, linewidth=box_linewidth,
                          edgecolor=box_color,
                          facecolor='none')
        axmatrix.add_patch(rect2)


def is_consecutive(arr):
    min_val = min(arr)
    max_val = max(arr)

    if max_val - min_val + 1 != len(arr):
        return False  # If the range isn't equal to the length, they can't be consecutive

    seen = set()
    for num in arr:
        if num in seen:
            return False  # Duplicate numbers mean they can't be strictly consecutive
        seen.add(num)

    return True

def as_distance_matrix(similarity_matrix):
    lls = [[1.0 - value for value in row] for row in similarity_matrix]
    lls = np.asarray(lls)
    return (lls + lls.transpose()) / 2


def maximal_matching(M):
    return csr_matrix(M)


def agglomerative_clustering(similarity_matrix, n_expected_clusters):
    distances = as_distance_matrix(similarity_matrix)
    model = AgglomerativeClustering(
        metric='precomputed',
        n_clusters=n_expected_clusters,
        linkage='complete'
    ).fit(distances)
    cluster_assignment = [set() for _ in range(n_expected_clusters)]
    for i, cluster in zip(range(len(similarity_matrix)), model.labels_):
        cluster_assignment[cluster].add(i)

    return cluster_assignment, model, np.array(similarity_matrix)


# K-Means clustering could not be used, as it is impossible to determine the coordinates out from the distance matrix
# - I can use the inference of the points given their distances
# - But this always assumes that distances are valid for triangular inequality, and that similarities are always symmetrical, which is not the case for logical elements

def matrix_init_normalize(matrix, normalization):
    if normalization == "simple_laplacian":
        return csr_matrix(matrix - laplacian_diag(matrix))
    elif normalization == "sym_normalized_laplacian":
        d = laplacian_diag(matrix)
        sqrt = np.sqrt(d)
        I = np.identity(matrix.shape[0])
        return csr_matrix(I - sqrt * matrix * sqrt)
    elif normalization == "random_walk_normalized":
        d = laplacian_diag(matrix)
        matrix = np.reciprocal(d, where=d != 0) * matrix
        I = np.identity(matrix.shape[0])
        return csr_matrix(I - matrix)
    else:
        return csr_matrix(matrix)


def knn(similarity_matrix, n_expected_clusters):
    distances = as_distance_matrix(similarity_matrix)
    from sklearn_extra.cluster import KMedoids
    model = KMedoids(n_clusters=n_expected_clusters, metric="precomputed", init="k-medoids++").fit(distances)
    cluster_assignment = [set() for _ in range(n_expected_clusters)]
    for i, cluster in zip(range(len(similarity_matrix)), model.labels_):
        cluster_assignment[cluster].add(i)
    return cluster_assignment, model, np.array(similarity_matrix)


def mcl_clustering_matches(similarity_matrix, expected_clusters):
    normalization = ["simple_laplacian", "sym_normalized_laplacian", "random_walk_normalized", "none"]
    modularity = 1000000
    candidate_result = None
    best_inflation = None
    best_norm = None
    for norm in normalization:
        matrix = csr_matrix(similarity_matrix)
        matrix = matrix_init_normalize(matrix, norm)
        # perform clustering using different inflation values from 1.5 and 2.5
        # for each clustering run, calculate the modularity
        for inflation in [i / 10 for i in range(15, 26)]:
            result = mc.run_mcl(matrix, inflation=inflation, loop_value=0)
            clusters = mc.get_clusters(result)
            Q = best_clustering_match([set(x) for x in clusters], expected_clusters)
            # Q = mc.modularity(matrix=result, clusters=clusters)
            if Q < modularity:
                print("inflation:", inflation, "modularity:", Q, "normalization:", norm)
                modularity = Q
                candidate_result = clusters
                best_inflation = inflation
                best_norm = norm
    assert modularity != -1000000
    assert candidate_result is not None

    print(f"Best Inflation: {best_inflation}")
    print(f"Best Normalization: {best_norm}")
    return [set(x) for x in clusters], matrix, candidate_result, best_inflation, best_norm


def set_matching_distance(X: set, Y: set):
    return (len(X.difference(Y)) + len(Y.difference(X))) / (len(X.union(Y)))


def best_clustering_match(minedClusters, expectedClusters):
    ## Assumptions: all the clusters are targeting non-overlaps, as sentence equivalence is transitive. thus, all the equivalent sentences shall belong to the same clsuter.
    minedClusters = [set(x) for x in minedClusters]
    expectedClusters = [set(x) for x in expectedClusters]
    total_alignment_score = 0
    matched_mined_clusters = set()
    for cluster in expectedClusters:
        result = -1
        score = 1
        for idx, x in enumerate(minedClusters):
            if idx not in matched_mined_clusters:
                d = set_matching_distance(x, cluster)
                if (d < score):
                    score = d
                    result = idx
        if result != -1:
            matched_mined_clusters.add(result)
            total_alignment_score += score
        if len(matched_mined_clusters) == len(expectedClusters):
            break
    unmatched_clusters = abs(len(expectedClusters) - len(matched_mined_clusters))
    # assert unmatched_clusters >= 0
    return (total_alignment_score + unmatched_clusters) / len(expectedClusters)


def dimsum(matrix, row=True):
    return matrix.sum(axis=1 if row else 0)


def laplacian_diag(matrix):
    return np.squeeze(np.asarray(dimsum(matrix, row=True)))
    # return numpy.diag(dimsum(matrix, row=True))


def matrix_exp2(matrix):
    return np.multiply(matrix, matrix)


def test_with_maximal_matching(expected_clusters, experiment_name, transformer, similarity_matrix=None,
                               implication_matrix=None):

    if similarity_matrix is None:
        similarity_matrix = read_json_array(
            f"catabolites/{experiment_name}/confusion_matrices_{transformer}.json")
        if similarity_matrix is None:
            return

    if transformer == "FullText_all-MiniLM-L6-v2":
        transformer = f"T1_{transformer}"
    elif transformer == "FullText_all-MiniLM-L12-v2":
        transformer = f"T2_{transformer}"
    elif transformer == "FullText_all-mpnet-base-v2":
        transformer = f"T3_{transformer}"
    elif transformer == "FullText_all-roberta-large-v1":
        transformer = f"T4_{transformer}"
    elif transformer == "FullText_AMR-LE-DeBERTa-V2-XXLarge-Contraposition-Double-Negation-Implication-Commutative-Pos-Neg-1-3":
        transformer = f"T5_{transformer}"
    elif transformer == "FullText_colbertv2.0":
        transformer = f"T6_{transformer}"
    # elif transformer == "SimpleGraph":
    #     transformer = "1SimpleGraph"
    # elif transformer == "LogicalGraph":
    #     transformer = "2LogicalGraph"
    # elif transformer == "Logical":
    #     transformer = "3Logical"

    row_name = f"{experiment_name}_{transformer}"

    not_implying_score = 0.0
    N = len(similarity_matrix)
    expected_labels = None
    roc_expected = None
    if implication_matrix is not None:
        expected_labels = []
        roc_expected = []
        for i in range(N):
            row = similarity_matrix[i]
            for j in range(N):
                cell = row[j]
                if cell == 1.0:
                    expected_labels.append(1)
                    roc_expected.append([1.0, 0.0, 0.0])
                elif cell == 0.0:
                    expected_labels.append(-1)
                    roc_expected.append([0.0, 0.0, 1.0])
                else:
                    expected_labels.append(0)
                    roc_expected.append([0.0, 1.0, 0.0])
        import itertools
        scores = list(itertools.chain.from_iterable(implication_matrix))
        n_not_implying = sum(1 for x in scores if x == 0.0)
        scores = list(itertools.chain.from_iterable(similarity_matrix))
        scores.sort()
        not_implying_score = max(scores[: n_not_implying])

    if not os.path.exists(f"catabolites/{experiment_name}"):
        os.makedirs(f"catabolites/{experiment_name}")

    if 'alice_bob' in experiment_name:
        box_clusters = None
    elif 'cat_mouse' in experiment_name:
        box_clusters = [2,3]
    elif 'newcastle' in experiment_name:
        box_clusters = [0,1,9]
    else:
        box_clusters = None

    print("Metrics (Agglomerative clustering)")
    n_expected_clusters = len(expected_clusters)
    agg_cluster_assignment, agg_model, distances = agglomerative_clustering(similarity_matrix, n_expected_clusters)
    agg_scores, implying_vs_indifferent, roc_scores = prepare_for_classical_clustering_metrics(
        N, agg_cluster_assignment, not_implying_score, similarity_matrix)
    print_metrics(agg_scores, expected_labels, implying_vs_indifferent, "Agglomerative", row_name)

    plot_dendogram(agg_model, distances, f"catabolites/{experiment_name}/{transformer}_dend.png", box_clusters)

    print("Metrics (Markov clustering)")
    mkv_cluster_assignment, matrix, mkv_clusters = knn(similarity_matrix, n_expected_clusters)
    agg_scores, implying_vs_indifferent, roc_scores = prepare_for_classical_clustering_metrics(
        N, mkv_cluster_assignment, not_implying_score, similarity_matrix)
    print_metrics(agg_scores, expected_labels, implying_vs_indifferent, "k-Medoids", row_name)

    agg_score = best_clustering_match(agg_cluster_assignment, expected_clusters)
    agg_similarity = 1 - agg_score
    print(f"Best Clustering Match (Agglomerative Clustering) [Proposed] Alignment Score: "
          f"{agg_similarity}. {agg_cluster_assignment}")
    metrics_benchmark.add_row(row_name, f"Best Clustering Match (Agglomerative Clustering) [Proposed] Alignment Score", agg_similarity)

    mkv_score = best_clustering_match(mkv_cluster_assignment, expected_clusters)
    mkv_similarity = 1 - mkv_score
    print(f"Best Clustering Match (k-Medoids) [Proposed] Alignment Score: {mkv_similarity}. {mkv_cluster_assignment}")
    metrics_benchmark.add_row(row_name, f"Best Clustering Match (k-Medoids) [Proposed] Alignment Score", mkv_similarity)

    expected_clusters_labels = get_labels(expected_clusters)
    print_silhouette_score(expected_clusters_labels, similarity_matrix, "Expected", row_name)

    agg_clusters_labels = get_labels(agg_cluster_assignment)
    print_silhouette_score(agg_clusters_labels, similarity_matrix, "Agglomerative Clustering", row_name)
    print(f"RND score (Agglomerative Clustering): {adjusted_rand_score(expected_clusters_labels, agg_clusters_labels)}")
    metrics_benchmark.add_row(row_name, f"RND score (Agglomerative Clustering)", adjusted_rand_score(expected_clusters_labels, agg_clusters_labels))
    print(f"Purity (Agglomerative Clustering): {purity_score(expected_clusters_labels, agg_clusters_labels)}")
    metrics_benchmark.add_row(row_name, f"Purity (Agglomerative Clustering)", purity_score(expected_clusters_labels, agg_clusters_labels))

    knn_clusters_labels = get_labels(mkv_cluster_assignment)
    print_silhouette_score(knn_clusters_labels, similarity_matrix, "k-Medoids", row_name)
    print(f"RND score (k-Medoids): {adjusted_rand_score(expected_clusters_labels, knn_clusters_labels)}")
    metrics_benchmark.add_row(row_name, f"RND score (k-Medoids)", mkv_similarity)
    print(f"Purity (k-Medoids): {purity_score(expected_clusters_labels, knn_clusters_labels)}")
    metrics_benchmark.add_row(row_name, f"Purity (k-Medoids)", mkv_similarity)


def print_silhouette_score(cluster_labels, similarity_matrix, type, row_name):
    try:
        calculated_silhouette_score = silhouette_score(as_distance_matrix(similarity_matrix), cluster_labels, metric='precomputed') if len(
            np.unique(cluster_labels)) != len(cluster_labels) else "N/A"
        print(f"Silhouette score ({type}): "
          f"{calculated_silhouette_score}")
        metrics_benchmark.add_row(row_name, f"Silhouette score ({type})", calculated_silhouette_score)
    except ValueError as e:
        X = as_distance_matrix(similarity_matrix)
        np.fill_diagonal(X, 0)  # TODO: Is it okay to do this?
        calculated_silhouette_score = silhouette_score(X, cluster_labels,
                                                       metric='precomputed') if len(
            np.unique(cluster_labels)) != len(cluster_labels) else "N/A"
        print(f"Silhouette score ({type}): {calculated_silhouette_score}")
        metrics_benchmark.add_row(row_name, f"Silhouette score ({type})", calculated_silhouette_score)


def print_metrics(agg_scores, expected_labels, implying_vs_indifferent, type, row_name):
    if expected_labels is not None and agg_scores is not None:
        print(f"Threshold value ({type}): "
              f"{implying_vs_indifferent}")
        # metrics_benchmark.add_row(row_name, f"Threshold value ({type})", implying_vs_indifferent)

        print(f"Accuracy Score ({type}): "
              f"{accuracy_score(expected_labels, agg_scores)}")
        metrics_benchmark.add_row(row_name, f"Accuracy Score ({type})", accuracy_score(expected_labels, agg_scores))

        print(f"Macro-F1 Score ({type}): "
              f"{f1_score(expected_labels, agg_scores, average='macro')}")
        metrics_benchmark.add_row(row_name, f"Macro-F1 Score ({type})", f1_score(expected_labels, agg_scores, average='macro'))

        print(f"Weighted-F1 Score ({type}): "
              f"{f1_score(expected_labels, agg_scores, average='weighted')}")
        metrics_benchmark.add_row(row_name, f"Weighted-F1 Score ({type})", f1_score(expected_labels, agg_scores, average='weighted'))

        print(f"Macro-Precision Score ({type}): "
              f"{precision_score(expected_labels, agg_scores, average='macro')}")
        metrics_benchmark.add_row(row_name, f"Macro-Precision Score ({type})", precision_score(expected_labels, agg_scores, average='macro'))

        print(f"Weighted-Precision Score ({type}): "
              f"{precision_score(expected_labels, agg_scores, average='weighted')}")
        metrics_benchmark.add_row(row_name, f"Weighted-Precision Score ({type})", precision_score(expected_labels, agg_scores, average='weighted'))

        print(f"Macro-Recall Score ({type}): "
              f"{recall_score(expected_labels, agg_scores, average='macro')}")
        metrics_benchmark.add_row(row_name, f"Macro-Recall Score ({type})", recall_score(expected_labels, agg_scores, average='macro'))

        print(f"Weighted-Recall Score ({type}): "
              f"{recall_score(expected_labels, agg_scores, average='weighted')}")
        metrics_benchmark.add_row(row_name, f"Weighted-Recall Score ({type})", recall_score(expected_labels, agg_scores, average='weighted'))


def prepare_for_classical_clustering_metrics(N, agg_cluster_assignment, not_implying_score, similarity_matrix):
    implying_vs_indifferent = sys.float_info.max
    for cluster in agg_cluster_assignment:
        for j in cluster:
            for i in cluster:
                implying_vs_indifferent = min([implying_vs_indifferent, similarity_matrix[i][j]])
    if implying_vs_indifferent <= not_implying_score:
        implying_vs_indifferent = not_implying_score
    # assert implying_vs_indifferent > not_implying_score
    agg_scores = []
    roc_scores = []
    idx = 0
    for i in range(N):
        row = similarity_matrix[i]
        for j in range(N):
            cell = row[j]
            implying_score = 0.0
            indifferent_score = 1.0
            wrong_score = 0.0
            if cell >= implying_vs_indifferent:
                agg_scores.append(1)
                implying_score = 1.0
                indifferent_score = 0.0 if (implying_vs_indifferent == cell) else 1.0 - (implying_vs_indifferent - cell)
                wrong_score = 0.0 if (not_implying_score == cell) else 1.0 - abs(cell - not_implying_score)
                roc_scores.append([implying_score, indifferent_score, wrong_score])
            elif cell <= not_implying_score:
                agg_scores.append(-1)
                implying_score = 0.0 if (implying_vs_indifferent == cell) else 1.0 - (implying_vs_indifferent - cell)
                indifferent_score = 0.0 if (not_implying_score == cell) else 1.0 - abs(cell - not_implying_score)
                wrong_score = 1.0
                roc_scores.append([implying_score, indifferent_score, wrong_score])
            else:
                agg_scores.append(0)
                implying_score = 0.0 if (implying_vs_indifferent == cell) else 1.0 - (implying_vs_indifferent - cell)
                indifferent_score = 1.0
                wrong_score = 0.0 if (not_implying_score == cell) else 1.0 - (cell - not_implying_score)
                roc_scores.append([implying_score, indifferent_score, wrong_score])
            idx += 1
    return agg_scores, implying_vs_indifferent, roc_scores


def get_labels(expected_clusters):
    N = 0
    d = dict()
    for idx, cl in enumerate(expected_clusters):
        for x in cl:
            d[x] = idx
            N += 1
    L = [d[x] for x in range(N)]
    return L


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def read_json_array(filepath):
    try:
        root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
        full_filepath = os.path.join(root_dir, filepath)

        with open(full_filepath, 'r') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


if __name__ == '__main__':
    tests = [
        [[[0], [1], [2], [3], [4], [5], [6], [7]], "alice_bob"],
        [[[0, 1], [2, 3], [4], [5]], "cat_mouse"],
        [[[0, 1, 9], [2], [3], [4], [5], [6, 7, 8], [10], [11], [12]], "newcastle_mdpi"]
    ]
    transformers = ["SimpleGraph", "LogicalGraph", "Logical", "FullText_all-MiniLM-L6-v2", "FullText_all-MiniLM-L12-v2",
                    "FullText_all-mpnet-base-v2", "FullText_all-roberta-large-v1"]

    for test in tests:
        print(test[1])
        for transformer in transformers:
            print(transformer)
            test_with_maximal_matching(test[0], test[1], transformer)
            print("\n")
