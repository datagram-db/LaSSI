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
        count += 1
        # print(count)
        if count % 1000 == 0: print(count)
        for j in adjacency_db.keys():

            if j not in adjacency_db[i]: continue

            # at this point, i -> j

            for k in adjacency_db.keys():

                if k not in adjacency_db[j]: continue
                if i == k: continue # this prevents self loops (build_clusters accomplishes this)

                # at this point, j -> k, hence i -> j -> k

                if k in adjacency_db[i]: continue  ## not completely sure if this is necessary, it's to guard against duplicates
                adjacency_list = adjacency_db[i]
                adjacency_list.append(k)
                adjacency_db[i] = adjacency_list  # i -> k
                # a later iteration will make this i <-> k (bijective)



def build_closures():
    adjacency_db = SqliteDict("adjacency_list.db")

    print("expanding adjacencny list with transitive closure")
    floyd_warshall(adjacency_db)

    adjacency_db.commit()
    adjacency_db.close()


def build_clusters():
    cluster_db = SqliteDict("clusters.db")
    adjacency_db = SqliteDict("adjacency_list.db")

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

    cluster_db.commit()
    cluster_db.close()
    adjacency_db.close()


def get_node(node):
    with SqliteDict("clusters.db") as cluster_db:
        cluster = cluster_db.get(node)
        return cluster[0] if cluster else node
