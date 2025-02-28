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


def test_with_maximal_matching(similarity_matrix, expected_clusters, experiment_name, transformer):
    print("Agglomerative clustering")
    n_expected_clusters = len(expected_clusters)
    agg_cluster_assignment, agg_model, distances = agglomerative_clustering(similarity_matrix, n_expected_clusters)
    plot_dendogram(agg_model, distances, f"{experiment_name}_{transformer}_dend.png")

    print("Markov clustering")
    mkv_cluster_assignment, matrix, mkv_clusters, best_inflation, best_norm = mcl_clustering_matches(similarity_matrix, expected_clusters)
    graph_plot(matrix, mkv_clusters, f"{experiment_name}_{transformer}_mkv.png")

    agg_score = best_clustering_match(agg_cluster_assignment, expected_clusters)
    agg_similarity = 1-agg_score
    print(f"Best Clustering Match (Agglomerative Clustering): {agg_similarity}. {agg_cluster_assignment}")

    mkv_score = best_clustering_match(mkv_cluster_assignment, expected_clusters)
    mkv_similarity = 1-mkv_score
    print(f"Best Clustering Match (Markov Clustering): {mkv_similarity}. {mkv_cluster_assignment}")


if __name__ == '__main__':
    # AB
    print("Alice Bob")
    expected = [[0], [1], [2], [3], [4], [5], [6], [7]]
    # Logical
    similarities = [[1.0, 0.5, 0.5, 1.0, 0.0, 0.5, 0.5, 0.5], [0.5, 1.0, 0.5, 1.0, 0.5, 0.0, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.0], [0.6666666666666666, 0.6666666666666666, 0.3333333333333333, 1.0, 0.3333333333333333, 0.3333333333333333, 0.5, 0.6666666666666666], [0.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.5, 1.0], [0.5, 0.0, 0.0, 0.5, 0.5, 1.0, 0.5, 1.0], [0.5, 0.5, 0.25, 0.75, 0.5, 0.5, 1.0, 0.75], [0.3333333333333333, 0.3333333333333333, 0.0, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.5, 1.0]]
    test_with_maximal_matching(similarities, expected, "alice_bob", "LaSSI")

    # L6
    similarities = [[1.0, 0.5816650390625, 0.8461833000183105, 0.8295102119445801, 0.8972232937812805, 0.547532320022583, 0.4502581059932709, 0.7521922588348389], [0.5816650390625, 1.0, 0.7301595211029053, 0.6947897672653198, 0.5195860266685486, 0.875572681427002, 0.5781753063201904, 0.6304416656494141], [0.8461833000183105, 0.7301595211029053, 1.0, 0.8888888888888888, 0.7937139868736267, 0.6924098134040833, 0.3894897699356079, 0.8543826937675476], [0.8295102119445801, 0.6947897672653198, 0.8888888888888888, 1.0, 0.7937861680984497, 0.6806595921516418, 0.39167922735214233, 0.8769489526748657], [0.8972232937812805, 0.5195860266685486, 0.7937139868736267, 0.7937861680984497, 1.0, 0.6802270412445068, 0.43657028675079346, 0.812751054763794], [0.547532320022583, 0.875572681427002, 0.6924098134040833, 0.6806595921516418, 0.6802270412445068, 1.0, 0.5470712780952454, 0.7383895516395569], [0.4502581059932709, 0.5781753063201904, 0.3894897699356079, 0.39167922735214233, 0.43657028675079346, 0.5470712780952454, 1.0, 0.3664126694202423], [0.7521922588348389, 0.6304416656494141, 0.8543826937675476, 0.8769489526748657, 0.812751054763794, 0.7383895516395569, 0.3664126694202423, 1.0]]
    test_with_maximal_matching(similarities, expected, "alice_bob", "all-MiniLM-L6-v2")

    # L12
    similarities = [[1.0, 0.5471906065940857, 0.8348720669746399, 0.8530226349830627, 0.9023645520210266, 0.5154948830604553, 0.5152065753936768, 0.7108111381530762], [0.5471906065940857, 1.0, 0.750140905380249, 0.7454906702041626, 0.49153026938438416, 0.8538334965705872, 0.6016612648963928, 0.5658745765686035], [0.8348720669746399, 0.750140905380249, 1.0, 0.8888888888888888, 0.7434138655662537, 0.6658003330230713, 0.5037853121757507, 0.7953468561172485], [0.8530226349830627, 0.7454906702041626, 0.8888888888888888, 1.0, 0.7894268035888672, 0.6991230249404907, 0.5233330726623535, 0.8282629251480103], [0.9023645520210266, 0.49153026938438416, 0.7434138655662537, 0.7894268035888672, 1.0, 0.6630093455314636, 0.5088055729866028, 0.7997223138809204], [0.5154948830604553, 0.8538334965705872, 0.6658003330230713, 0.6991230249404907, 0.6630093455314636, 1.0, 0.5917628407478333, 0.7067323923110962], [0.5152065753936768, 0.6016612648963928, 0.5037853121757507, 0.5233330726623535, 0.5088055729866028, 0.5917628407478333, 1.0, 0.42749133706092834], [0.7108111381530762, 0.5658745765686035, 0.7953468561172485, 0.8282629251480103, 0.7997223138809204, 0.7067323923110962, 0.42749133706092834, 1.0]]
    test_with_maximal_matching(similarities, expected, "alice_bob", "all-MiniLM-L12-v2")

    # mpnet
    similarities = [[1.0, 0.5814658403396606, 0.8217214345932007, 0.7869688272476196, 0.8787805438041687, 0.5412308573722839, 0.5218508243560791, 0.7228233218193054], [0.5814658403396606, 1.0, 0.7717898488044739, 0.7280459403991699, 0.5189411044120789, 0.8722764849662781, 0.5905681848526001, 0.6531456708908081], [0.8217214345932007, 0.7717898488044739, 1.0, 0.8888888888888888, 0.7151430249214172, 0.6979902982711792, 0.5063621997833252, 0.8108426928520203], [0.7869688272476196, 0.7280459403991699, 0.8888888888888888, 1.0, 0.7195632457733154, 0.6780077815055847, 0.4990215599536896, 0.832362949848175], [0.8787805438041687, 0.5189411044120789, 0.7151430249214172, 0.7195632457733154, 1.0, 0.677765965461731, 0.5070024728775024, 0.842663049697876], [0.5412308573722839, 0.8722764849662781, 0.6979902982711792, 0.6780077815055847, 0.677765965461731, 1.0, 0.5671989321708679, 0.7873305678367615], [0.5218508243560791, 0.5905681848526001, 0.5063621997833252, 0.4990215599536896, 0.5070024728775024, 0.5671989321708679, 1.0, 0.4767208695411682], [0.7228233218193054, 0.6531456708908081, 0.8108426928520203, 0.832362949848175, 0.842663049697876, 0.7873305678367615, 0.4767208695411682, 1.0]]
    test_with_maximal_matching(similarities, expected, "alice_bob", "all-mpnet-base-v2")

    # roberta
    similarities = [[1.0, 0.7208858132362366, 0.8081421256065369, 0.8476463556289673, 0.8632146120071411, 0.5944067239761353, 0.5625920295715332, 0.7191655039787292], [0.7208858132362366, 1.0, 0.7861891984939575, 0.7875993251800537, 0.553360641002655, 0.8468124866485596, 0.7019233107566833, 0.6705491542816162], [0.8081421256065369, 0.7861891984939575, 1.0, 0.8888888888888888, 0.6398048400878906, 0.6338595747947693, 0.5136296153068542, 0.8296692371368408], [0.8476463556289673, 0.7875993251800537, 0.8888888888888888, 1.0, 0.7371706962585449, 0.7151520848274231, 0.5607743263244629, 0.8260473012924194], [0.8632146120071411, 0.553360641002655, 0.6398048400878906, 0.7371706962585449, 1.0, 0.6764675378799438, 0.46038001775741577, 0.7734036445617676], [0.5944067239761353, 0.8468124866485596, 0.6338595747947693, 0.7151520848274231, 0.6764675378799438, 1.0, 0.6259873509407043, 0.7549120187759399], [0.5625920295715332, 0.7019233107566833, 0.5136296153068542, 0.5607743263244629, 0.46038001775741577, 0.6259873509407043, 1.0, 0.47645139694213867], [0.7191655039787292, 0.6705491542816162, 0.8296692371368408, 0.8260473012924194, 0.7734036445617676, 0.7549120187759399, 0.47645139694213867, 1.0]]
    test_with_maximal_matching(similarities, expected, "alice_bob", "all-roberta-large-v1")

    # CM
    print("CM")
    expected = [[0,1], [2,3], [4], [5]]
    # Logical
    similarities = [[1.0, 1.0, 0.5, 0.5, 0.0, 0.5], [1.0, 1.0, 0.5, 0.5, 0.0, 0.5], [0.5, 0.5, 1.0, 1.0, 0.5, 0.0], [0.5, 0.5, 1.0, 1.0, 0.5, 0.0], [0.0, 0.0, 0.5, 0.5, 1.0, 0.5], [0.5, 0.5, 0.0, 0.0, 0.5, 1.0]]
    test_with_maximal_matching(similarities, expected, "cat_mouse", "LaSSI")

    # L6
    similarities = [[1.0, 0.9217581152915955, 0.9718253158075502, 0.9415593147277832, 0.8528849482536316, 0.8184587955474854], [0.9217581152915955, 1.0, 0.9397860765457153, 0.9797958971132713, 0.8772938847541809, 0.8654747009277344], [0.9718253158075502, 0.9397860765457153, 1.0, 0.9457026124000549, 0.8722233772277832, 0.8602184653282166], [0.9415593147277832, 0.9797958971132713, 0.9457026124000549, 1.0, 0.869746744632721, 0.8445749282836914], [0.8528849482536316, 0.8772938847541809, 0.8722233772277832, 0.869746744632721, 1.0, 1.0], [0.8184587955474854, 0.8654747009277344, 0.8602184653282166, 0.8445749282836914, 1.0, 1.0]]
    test_with_maximal_matching(similarities, expected, "cat_mouse", "all-MiniLM-L6-v2")

    # L12
    similarities = [[1.0, 0.9592887163162231, 0.9718253158075502, 0.9566174745559692, 0.8173177242279053, 0.8363404870033264], [0.9592887163162231, 1.0, 0.9473628401756287, 0.9797958971132713, 0.7681660652160645, 0.7881879210472107], [0.9718253158075502, 0.9473628401756287, 1.0, 0.9386256337165833, 0.8273110389709473, 0.8630075454711914], [0.9566174745559692, 0.9797958971132713, 0.9386256337165833, 1.0, 0.7629190683364868, 0.7773761749267578], [0.8173177242279053, 0.7681660652160645, 0.8273110389709473, 0.7629190683364868, 1.0, 1.0], [0.8363404870033264, 0.7881879210472107, 0.8630075454711914, 0.7773761749267578, 1.0, 1.0]]
    test_with_maximal_matching(similarities, expected, "cat_mouse", "all-MiniLM-L12-v2")

    # mpnet
    similarities = [[1.0, 0.7726534605026245, 0.9718253158075502, 0.8103326559066772, 0.8619885444641113, 0.8708223700523376], [0.7726534605026245, 1.0, 0.8137537837028503, 0.9797958971132713, 0.691731333732605, 0.7037764191627502], [0.9718253158075502, 0.8137537837028503, 1.0, 0.8476861119270325, 0.8194189071655273, 0.8495117425918579], [0.8103326559066772, 0.9797958971132713, 0.8476861119270325, 1.0, 0.7134833335876465, 0.7239227294921875], [0.8619885444641113, 0.691731333732605, 0.8194189071655273, 0.7134833335876465, 1.0, 1.0], [0.8708223700523376, 0.7037764191627502, 0.8495117425918579, 0.7239227294921875, 1.0, 1.0]]
    test_with_maximal_matching(similarities, expected, "cat_mouse", "all-mpnet-base-v2")

    # roberta
    similarities = [[1.0, 0.8595811724662781, 0.9718253158075502, 0.858819305896759, 0.7979873418807983, 0.7956418991088867], [0.8595811724662781, 1.0, 0.839458703994751, 0.9797958971132713, 0.6915876269340515, 0.7025970816612244], [0.9718253158075502, 0.839458703994751, 1.0, 0.8623122572898865, 0.741420567035675, 0.791620671749115], [0.858819305896759, 0.9797958971132713, 0.8623122572898865, 1.0, 0.6902737617492676, 0.7177722454071045], [0.7979873418807983, 0.6915876269340515, 0.741420567035675, 0.6902737617492676, 1.0, 1.0], [0.7956418991088867, 0.7025970816612244, 0.791620671749115, 0.7177722454071045, 1.0, 1.0]]
    test_with_maximal_matching(similarities, expected, "cat_mouse", "all-roberta-large-v1")

    # Newcastle
    print("NEwcastle")
    expected = [[0, 1, 2], [3, 4, 5], [6], [7], [8]]
    # similarities = []
    # test_with_maximal_matching(similarities, expected, "newcastle", "LaSSI")

    # L6
    similarities = [[1.0, 0.9432422182837986, 0.914991421995628, 0.8808189034461975, 0.7633077502250671, 0.36761611700057983, 0.9014837741851807, 0.8806849122047424, 0.8901776075363159, 0.975773274898529, 0.8601517677307129, 0.9012788534164429, 0.9001208543777466, 0.8359494209289551], [0.9432422182837986, 1.0, 0.8630585385938034, 0.8624575450871764, 0.7464306950569153, 0.37896618247032166, 0.899112343788147, 0.8805673122406006, 0.8886292576789856, 0.973406195640564, 0.8622534871101379, 0.9027550220489502, 0.9028400182723999, 0.8327628970146179], [0.914991421995628, 0.8630585385938034, 1.0, 0.8167401552200317, 0.7184513807296753, 0.4090675711631775, 0.8314874172210693, 0.8097530603408813, 0.8175151348114014, 0.9054692983627319, 0.9072349667549133, 0.9288347410476049, 0.928702175617218, 0.7927631139755249], [0.8808189034461975, 0.8624575450871764, 0.8167401552200317, 1.0, 0.6527912020683289, 0.32727479934692383, 0.8282380700111389, 0.7926076054573059, 0.7966253161430359, 0.8824975032927699, 0.8291652202606201, 0.8430483937263489, 0.8409144878387451, 0.7833044528961182], [0.7633077502250671, 0.7464306950569153, 0.7184513807296753, 0.6527912020683289, 1.0, 0.45391595363616943, 0.6963629722595215, 0.6685981750488281, 0.6828809380531311, 0.7049856781959534, 0.6345646381378174, 0.7118881940841675, 0.6550906300544739, 0.6863353848457336], [0.36761611700057983, 0.37896618247032166, 0.4090675711631775, 0.32727479934692383, 0.45391595363616943, 1.0, 0.5851866602897644, 0.5645159482955933, 0.5802938938140869, 0.3957027196884155, 0.4065568447113037, 0.4089520573616028, 0.38321149349212646, 0.3979324698448181], [0.9014837741851807, 0.899112343788147, 0.8314874172210693, 0.8282380700111389, 0.6963629722595215, 0.5851866602897644, 1.0, 0.8680790595108567, 0.8765483240617118, 0.910306453704834, 0.837088406085968, 0.841884970664978, 0.8529934287071228, 0.781609296798706], [0.8806849122047424, 0.8805673122406006, 0.8097530603408813, 0.7926076054573059, 0.6685981750488281, 0.5645159482955933, 0.8680790595108567, 1.0, 0.9320827648567408, 0.879904568195343, 0.7754738926887512, 0.809771716594696, 0.8064208030700684, 0.7494145631790161], [0.8901776075363159, 0.8886292576789856, 0.8175151348114014, 0.7966253161430359, 0.6828809380531311, 0.5802938938140869, 0.8765483240617118, 0.9320827648567408, 1.0, 0.8878212571144104, 0.7816958427429199, 0.8170656561851501, 0.814659833908081, 0.7559835314750671], [0.975773274898529, 0.973406195640564, 0.9054692983627319, 0.8824975032927699, 0.7049856781959534, 0.3957027196884155, 0.910306453704834, 0.879904568195343, 0.8878212571144104, 1.0, 0.905209481716156, 0.9230144619941711, 0.9354211688041687, 0.8462367057800293], [0.8601517677307129, 0.8622534871101379, 0.9072349667549133, 0.8291652202606201, 0.6345646381378174, 0.4065568447113037, 0.837088406085968, 0.7754738926887512, 0.7816958427429199, 0.905209481716156, 1.0, 0.936873197555542, 0.9428090415820634, 0.7956444025039673], [0.9012788534164429, 0.9027550220489502, 0.9288347410476049, 0.8430483937263489, 0.7118881940841675, 0.4089520573616028, 0.841884970664978, 0.809771716594696, 0.8170656561851501, 0.9230144619941711, 0.936873197555542, 1.0, 0.9610179662704468, 0.7950783371925354], [0.9001208543777466, 0.9028400182723999, 0.928702175617218, 0.8409144878387451, 0.6550906300544739, 0.38321149349212646, 0.8529934287071228, 0.8064208030700684, 0.814659833908081, 0.9354211688041687, 0.9428090415820634, 0.9610179662704468, 1.0, 0.8086408376693726], [0.8359494209289551, 0.8327628970146179, 0.7927631139755249, 0.7833044528961182, 0.6863353848457336, 0.3979324698448181, 0.781609296798706, 0.7494145631790161, 0.7559835314750671, 0.8462367057800293, 0.7956444025039673, 0.7950783371925354, 0.8086408376693726, 1.0]]
    test_with_maximal_matching(similarities, expected, "newcastle", "all-MiniLM-L6-v2")
    #
    # # L12
    # similarities = []
    # test_with_maximal_matching(similarities, expected, "newcastle", "all-MiniLM-L12-v2")
    #
    # # mpnet
    # similarities = []
    # test_with_maximal_matching(similarities, expected, "newcastle", "all-mpnet-base-v2")
    #
    # # roberta
    # similarities = []
    # test_with_maximal_matching(similarities, expected, "newcastle", "all-roberta-large-v1")





