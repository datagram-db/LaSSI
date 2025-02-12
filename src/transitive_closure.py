from sqlitedict import SqliteDict

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
def build_closures():
    adjacency_db = SqliteDict('adjacency_list.db')

    for i in adjacency_db.keys():
        for j in adjacency_db.keys():

            if j not in adjacency_db[i]: continue

            # at this point, i -> j

            for k in adjacency_db.keys():

                if k not in adjacency_db[j]: continue

                # at this point, j -> k, hence i -> j -> k

                if k in adjacency_db[i]: continue ## not completely sure if this is necessary, it's to guard against duplicates
                adjacency_list = adjacency_db[i]
                adjacency_list.append(k)
                adjacency_db[i] = adjacency_list # i -> k
                # a later iteration will make this i <-> k (bijective)

    adjacency_db.commit()
    adjacency_db.close()


def build_clusters():
    cluster_db = SqliteDict('clusters.db')
    adjacency_db = SqliteDict('adjacency_list.db')

    for key_node, adjacency_list in adjacency_db.items():
        if cluster_db.get(key_node): continue

        adjacency_list.append(key_node)
        cluster = sorted(adjacency_list)

        for node in cluster:
            cluster_db[node] = cluster

    cluster_db.commit()
    cluster_db.close()
    adjacency_db.close()
