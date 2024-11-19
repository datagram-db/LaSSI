# Phase -1
def getJsonAsAdjacencyList(gsm_json):
    gsm_id_json = {row['id']: row for row in gsm_json}

    num_of_nodes = max(item['id'] for item in gsm_json) + 1
    adj = [[] for _ in range(num_of_nodes)]

    for i in gsm_id_json:
        node = gsm_id_json[i]
        if len(node['phi']) > 0:
            for edge in node['phi']:
                adj[edge['score']['parent']].append(edge['score']['child'])

    return adj, num_of_nodes


def topologicalSortUtil(v, adj, visited, stack):
    # Mark the current node as visited
    visited[v] = True

    # Recur for all adjacent vertices
    for i in adj[v]:
        if not visited[i]:
            topologicalSortUtil(i, adj, visited, stack)

    # Push current vertex to stack which stores the result
    stack.append(v)


def topologicalSort(gsm_json):
    adj, num_of_nodes = getJsonAsAdjacencyList(gsm_json)

    stack = []
    visited = [False] * num_of_nodes

    for i in range(num_of_nodes):
        if not visited[i]:
            topologicalSortUtil(i, adj, visited, stack)

    json_ids = {item['id'] for item in gsm_json}

    # Create a dictionary mapping IDs to their positions in the stack,
    # but only for IDs that exist in the JSON data
    id_order = {id: index for index, id in enumerate(stack) if id in json_ids}
    return sorted(gsm_json, key=lambda item: id_order.get(item['id']))