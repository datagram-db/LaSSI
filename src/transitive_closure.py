from sqlitedict import SqliteDict

def build_clusters():
    """
    nodes = adj_db.getNodes()

    initializing clusters for each node, that just includes said node
    for n in nodes:
        id = new_id()
        cluster_members_db[id] = [n]
        cluster_mapping_db[n] = id

    it's gonna be like:
    for i in nodes:
    for j in nodes:
    for k in nodes:
    if i == j and j == k:
    do cluster thing
    """
    adjacency_db = SqliteDict('adjacency_list.db')
    cluster_db = SqliteDict('clusters.db')

    for

