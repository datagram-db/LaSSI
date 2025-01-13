__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2024, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

import numpy
from matplotlib import pylab
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import csr_matrix
import markov_clustering as mc
import networkx as nx
import matplotlib
import matplotlib.pyplot

def graph_plot(matrix, clusters, filename="graph.png"):
    fig = matplotlib.pyplot.figure()
    ## Color blind palette: https://github.com/mpetroff/accessible-color-cycles
    palette = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
    assert len(clusters) <= len(palette)
    graph = nx.Graph(matrix)
    cluster_map = {node: i for i, cluster in enumerate(clusters) for node in cluster}
    edges, weights = zip(*nx.get_edge_attributes(graph, 'weight').items())
    colors = [palette[cluster_map[i]] for i in range(len(graph.nodes()))]
    positions = nx.spring_layout(graph, seed=31) ## ensuring the same layout for the same graph!
    nx.draw_networkx(graph, node_color=colors, pos=positions, with_labels=True,  edgelist=edges, edge_color=weights, edge_cmap=matplotlib.cm.YlGnBu, ax=fig.add_subplot())
    if filename is not None:
        # Save plot to file
        matplotlib.use("Agg")
        fig.savefig(filename)
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

    import numpy as np
    from scipy.cluster.hierarchy import dendrogram
    # Children of hierarchical clustering
    children = model.children_
    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])
    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)
    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, **kwargs)

def plot_dendogram(model, D, filename="dendrogram.png"):
    # fig = matplotlib.pyplot.figure()
    from scipy.spatial.distance import squareform
    # Compute and plot first dendrogram.
    # condensedD = squareform(D)
    import scipy.cluster.hierarchy as sch
    fig = matplotlib.pyplot.figure(figsize=(8, 8))
    ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6])
    # Y = sch.linkage(condensedD, method='centroid')
    Z1 = _plot_dendrogram(model, orientation='left')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
    # Y = sch.linkage(condensedD, method='single')
    Z2 = _plot_dendrogram(model)
    ax2.set_xticks([])
    ax2.set_yticks([])

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
    matplotlib.pyplot.colorbar(im, cax=axcolor)
    axmatrix.set_xticks(range(len(idx1)))
    axmatrix.set_xticklabels(idx1, minor=False)
    # axmatrix.xaxis.set_label_position('bottom')
    # axmatrix.xaxis.tick_bottom()
    #
    # pylab.xticks(rotation=-90, fontsize=8)
    #
    axmatrix.set_yticks(range(len(idx2)))
    axmatrix.set_yticklabels(idx2, minor=False)
    # axmatrix.yaxis.set_label_position('right')
    # axmatrix.yaxis.tick_right()

    # axcolor = fig.add_axes([0.94, 0.1, 0.02, 0.6])
    # plt.show()
    # _plot_dendrogram(model, labels=model.labels_, ax=fig.add_subplot())
    if filename is not None:
        # Save plot to file
        matplotlib.use("Agg")
        fig.savefig(filename)
    else:
        # Display interactive viewer
        matplotlib.pyplot.show()

def as_distance_matrix(similarity_matrix):
    return [[1.0-value for value in row] for row in similarity_matrix]



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

    return cluster_assignment, model, numpy.array(similarity_matrix)

# K-Means clustering could not be used, as it is impossible to determine the centroids out from the distance matrix

def matrix_init_normalize(matrix, normalization):
    import numpy
    if normalization == "simple_laplacian":
        return csr_matrix(matrix - laplacian_diag(matrix))
    elif normalization == "sym_normalized_laplacian":
        d = laplacian_diag(matrix)
        sqrt = numpy.sqrt(d)
        I = numpy.identity(matrix.shape[0])
        return csr_matrix(I - sqrt * matrix * sqrt)
    elif normalization == "random_walk_normalized":
        d = laplacian_diag(matrix)
        matrix = numpy.reciprocal(d,where= d!=0) * matrix
        I = numpy.identity(matrix.shape[0])
        return csr_matrix(I - matrix)
    else:
        return csr_matrix(matrix)

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


def set_matching_distance(X : set, Y : set):
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
    import numpy
    return numpy.squeeze(numpy.asarray(dimsum(matrix, row=True)))
    # return numpy.diag(dimsum(matrix, row=True))

def matrix_exp2(matrix):
    import numpy
    return numpy.multiply(matrix, matrix)


def test_with_maximal_matching(similarity_matrix, expected_clusters, experiment_name):
    print("Agglomerative clustering")
    n_expected_clusters = len(expected_clusters)
    agg_cluster_assignment, agg_model, distances = agglomerative_clustering(similarity_matrix, n_expected_clusters)
    plot_dendogram(agg_model, distances, f"{experiment_name}_dend.png")

    print("Markov clustering")
    mkv_cluster_assignment, matrix, mkv_clusters, best_inflation, best_norm = mcl_clustering_matches(similarity_matrix, expected_clusters)
    graph_plot(matrix, mkv_clusters, f"{experiment_name}_mkv.png")

    agg_score = best_clustering_match(agg_cluster_assignment, expected_clusters)
    agg_similarity = 1-agg_score
    print(f"Best Clustering Match (Agglomerative Clustering): {agg_similarity}. {agg_cluster_assignment}")

    mkv_score = best_clustering_match(mkv_cluster_assignment, expected_clusters)
    mkv_similarity = 1-mkv_score
    print(f"Best Clustering Match (Markov Clustering): {mkv_similarity}. {mkv_cluster_assignment}")


if __name__ == '__main__':
    similarities = [[1.0, 0, 0.5, 0.9], [0.3, 1.0, 0.0, 0.7], [0.0, 0.2, 1.0, 0.0], [0.7, 0.0, 0.0, 1.0]]
    expected = [[0,3], [2], [1]]
    test_with_maximal_matching(similarities, expected, "test")




