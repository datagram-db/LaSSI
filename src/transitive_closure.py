from sqlitedict import SqliteDict


def find(parent, node):
    if parent[node] != node:
        parent[node] = find(parent, parent[node])
    return parent[node]


def union(parent, node1, node2):
    root1 = find(parent, node1)
    root2 = find(parent, node2)

    # if roots r same then they were already merged
    if root1 != root2:
        parent[root2] = root1  # parenting one to the other, it doesnt matter which


def DSU(adjacency_db):

    parent = {node: node for node in adjacency_db.keys()}  # each node is its own parent

    for node in adjacency_db.keys():
        for neighbor in adjacency_db[node]:
            union(parent, node, neighbor)

    clusters = {}  # root -> set of connected nodes
    for node in adjacency_db.keys():
        root = find(parent, node)
        if root not in clusters:
            clusters[root] = set()
        clusters[root].add(node)   # the same root node can be added several times so Set type is useful

    for root, cluster in clusters.items():
        cluster_list = list(cluster)

        for node in cluster:
            adjacency_db[node] = [neighbor for neighbor in cluster_list if neighbor != node]  # if check excludes self loop



def floyd_warshall(adjacency_db):
    count = 0
    for i in adjacency_db.keys():
        adjacency_list = adjacency_db[i]
        count += 1
        j_idx = 0
        # print(count)
        if count % 1000 == 0: print(count)
        #for j in adjacency_db.keys():
        #for j in adjacency_list:
        while j_idx < len(adjacency_list):
            j = adjacency_list[j_idx]
            # if j not in adjacency_db[i]: continue

            # at this point, i -> j

            adjacency_list_j = adjacency_db[j]
            k_idx = 0
            # for k in adjacency_db.keys():
            while k_idx < len(adjacency_list_j):
                k = adjacency_list_j[k_idx]
                # if k not in adjacency_list_j: continue
                if i == k:
                    k_idx += 1
                    continue # this prevents self loops (build_clusters accomplishes this)

                # at this point, j -> k, hence i -> j -> k

                if k in adjacency_list:
                    k_idx += 1
                    continue  ## not completely sure if this is necessary, it's to guard against duplicates

                adjacency_list.append(k)
                k_idx += 1
            j_idx += 1
            adjacency_db[j] = adjacency_list_j
        adjacency_db[i] = adjacency_list  # i -> k
                # a later iteration will make this i <-> k (bijective)


def build_closures(adjacency_db):

    print("expanding adjacencny list with transitive closure")
    floyd_warshall(adjacency_db)

    if type(adjacency_db) == SqliteDict:
        adjacency_db.commit()


def build_clusters(adjacency_db, cluster_db):
    print("building clusters")
    count = 0
    for key_node, adjacency_list in adjacency_db.items():
        if cluster_db.get(key_node): continue

        count += 1
        if count % 1000 == 0: print(count)

        adjacency_list.append(key_node)
        cluster = sorted(adjacency_list)

        for node in cluster:
            cluster_db[node] = cluster

    if type(cluster_db) == SqliteDict:
        cluster_db.commit()


def get_node(cluster_db, node):
    cluster = cluster_db.get(node)
    return cluster[0] if cluster else node
