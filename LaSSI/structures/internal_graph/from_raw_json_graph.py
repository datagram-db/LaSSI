from LaSSI.structures.internal_graph.EntityRelationship import Singleton, Grouping, SetOfSingletons

from collections import defaultdict, deque
from itertools import repeat


def bfs(lists, s):
    nodes = {x['id']:x for x in lists}

    visited = set()

    # Create a queue for BFS
    q = deque()

    # Mark the source node as visited and enqueue it
    visited.add(s)
    q.append(s)

    # Iterate over the queue
    while q:

        # Dequeue a vertex from queue and print it
        id = q.popleft()

        if 'cc' in nodes[id]['properties']:
            return nodes[id]['properties']['cc']

        # Get all adjacent vertices of the dequeued
        # vertex. If an adjacent has not been visited,
        # mark it visited and enqueue it
        for edge in nodes[id]['phi']:
            dst = edge['score']['child']
            if dst not in visited:
                visited.add(dst)
                q.append(dst)

