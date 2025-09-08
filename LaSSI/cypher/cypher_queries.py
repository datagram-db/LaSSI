__author__ = "Oliver Robert Fox"
__copyright__ = "Copyright 2024, Oliver Robert Fox"
__credits__ = ["Oliver Robert Fox"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver Robert Fox"
__email__ = "ollie.fox5@gmail.com"
__status__ = "Production"

def create_node(id, name):
    node = {
        "type": "node",
        "id": id,
        "properties": {
            "name": name
        }
    }
    return node

def create_rel(id, name, start, end):
    rel = {
        "type": "relationship",
        "id": id,
        "label": name,
        "start": start,
        "end": end
    }
    return rel


def convert_to_cypher_json(input):
    lines = input.split("\n")

    # sim_graph = List[Relationship]

    i = 0
    nodes = []
    nodeIds = []
    rels = []
    for line in lines:
        split_start = line.split('--')  # Split into [start, rel-->end]
        if split_start != ['']:  # As long as valid relationship exists then
            split_end = split_start[1].split("->")  # Split into [rel, end]

            start_obj = split_start[0].replace('(', '').replace(')', '').split(',')
            startId = start_obj[0]
            start = start_obj[1]

            start_node = create_node(i, start)
            exists = False
            for node in nodeIds:
                if node[0] == startId:
                    start_node = node[1]
                    exists = True
                    break

            if not exists:
                nodes.append(start_node)
                nodeIds.append([startId, start_node])
            i += 1

            end_obj = split_end[1].replace('(', '').replace(')', '').split(',')
            end_id = end_obj[0]
            end = end_obj[1]

            end_node = create_node(i, end)
            exists = False
            for node in nodeIds:
                if node[0] == end_id:
                    end_node = node[1]
                    exists = True
                    break

            if not exists:
                nodes.append(end_node)
                nodeIds.append([end_id, end_node])
            i += 1

            rel = split_end[0].replace('[', '').replace(']', '')  # Remove brackets from rel object
            rel_obj = create_rel(i, rel, start_node, end_node)
            rels.append(rel_obj)
            i += 1

    return {
        "nodes": nodes,
        "rels": rels
    }
