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

    return ''

def group_nodes(nodes, parsed_json, simplsitic, parmenides):
    # Get all nodes from resulting graph and create list of Singletons
    number_of_nodes = range(len(parsed_json))
    for row in number_of_nodes:
        item = parsed_json[row]
        if 'conj' in item['properties'] or 'multipleindobj' in item['ell']:
            continue  # add set of singletons later as we might not have all nodes yet
        else:
            minV = -1
            maxV = -1
            typeV = "None"
            if len(item['xi']) > 0:
                name = item['xi'][0]
                minV = int(item['properties']['begin'])
                maxV = int(item['properties']['end'])
                typeV = item['ell'][0] if len(item['ell']) > 0 else "None"
            else:
                name = '?'  # xi might be empty if the node is invented

            nodes[item['id']] = Singleton(
                id=item['id'],
                named_entity=name,
                properties=frozenset(item['properties'].items()),
                min=minV,
                max=maxV,
                type=typeV,
                confidence=1.0
            )
    # Check if conjugation ('conj') exists and if true exists, merge into SetOfSingletons
    # Also if 'compound' relationship is present, merge parent and child nodes
    for row in number_of_nodes:
        item = parsed_json[row]
        grouped_nodes = []
        has_conj = 'conj' in item['properties']
        has_multipleindobj = 'multipleindobj' in item['ell']
        is_compound = False
        group_type = None
        norm_confidence = 1.0
        if has_conj:
            conj = item['properties']['conj'].strip()
            if len(conj) == 0:
                conj = bfs(parsed_json, item['id'])

            if 'and' in conj or 'but' in conj:
                group_type = Grouping.AND
            elif ('nor' in conj) or ('neither' in conj):
                group_type = Grouping.NEITHER
            elif 'or' in conj:
                group_type = Grouping.OR
            else:
                group_type = Grouping.NONE
            for edge in item['phi']:
                if 'orig' in edge['containment']:
                    node = nodes[edge['content']]
                    grouped_nodes.append(node)
                    norm_confidence *= node.confidence
        elif has_multipleindobj:
            for edge in item['phi']:
                if 'orig' in edge['containment']:
                    node = nodes[edge['content']]
                    grouped_nodes.append(node)
                    norm_confidence *= node.confidence
        else:
            for edge in item['phi']:
                is_current_edge_compound = 'compound' in edge['containment']
                if is_current_edge_compound:
                    is_compound = True
                    node = nodes[edge['score']['child']]
                    grouped_nodes.append(node)
                    norm_confidence *= node.confidence

        if simplsitic and len(grouped_nodes) > 0:
            sorted_entities = sorted(grouped_nodes, key=lambda x: float(dict(x.properties)['pos']))
            sorted_entity_names = list(map(getattr, sorted_entities, repeat('named_entity')))

            all_types = list(map(getattr, sorted_entities, repeat('type')))
            specific_type = parmenides.most_specific_type(all_types)

            if group_type == Grouping.OR:
                name = " or ".join(sorted_entity_names)
            elif group_type == Grouping.AND:
                name = " and ".join(sorted_entity_names)
            elif group_type == Grouping.NEITHER:
                name = " nor ".join(sorted_entity_names)
                name = f"neither {name}"
            else:
                name = " ".join(sorted_entity_names)

            nodes[item['id']] = Singleton(
                id=item['id'],
                named_entity=name,
                properties=frozenset(item['properties'].items()),
                min=min(grouped_nodes, key=lambda x: x.min).min,
                max=max(grouped_nodes, key=lambda x: x.max).max,
                type=specific_type,
                confidence=norm_confidence
            )
        elif not simplsitic:
            if has_conj:
                if group_type == Grouping.NEITHER:
                    grouped_nodes = [SetOfSingletons(id=x.id, type=Grouping.NOT, entities=tuple([x]), min=x.min, max=x.max,
                                                     confidence=x.confidence) for x in grouped_nodes]
                    grouped_nodes = tuple(grouped_nodes)
                    group_type = Grouping.AND
                nodes[item['id']] = SetOfSingletons(
                    id=item['id'],
                    type=group_type,
                    entities=tuple(grouped_nodes),
                    min=min(grouped_nodes, key=lambda x: x.min).min,
                    max=max(grouped_nodes, key=lambda x: x.max).max,
                    confidence=norm_confidence
                )
            elif is_compound:
                grouped_nodes.insert(0, nodes[item['id']])
                nodes[item['id']] = SetOfSingletons(
                    id=item['id'],
                    type=Grouping.GROUPING,
                    entities=tuple(grouped_nodes),
                    min=min(grouped_nodes, key=lambda x: x.min).min,
                    max=max(grouped_nodes, key=lambda x: x.max).max,
                    confidence=norm_confidence
                )
            elif has_multipleindobj:
                nodes[item['id']] = SetOfSingletons(
                    id=item['id'],
                    type=Grouping.MULTIINDIRECT,
                    entities=tuple(grouped_nodes),
                    min=min(grouped_nodes, key=lambda x: x.min).min,
                    max=max(grouped_nodes, key=lambda x: x.max).max,
                    confidence=norm_confidence
                )
