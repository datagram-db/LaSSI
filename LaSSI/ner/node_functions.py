from collections import deque, defaultdict

from LaSSI.external_services.Services import Services
from LaSSI.structures.internal_graph.EntityRelationship import Singleton, Grouping


class NodeFunctions:
    def __init__(self, max_id):
        self.services = Services.getInstance()
        self.existentials = self.services.getExistentials()
        self.node_id_map = dict()
        self.max_id = max_id

    def get_gsm_item_from_id(self, gsm_id, gsm_json):
        gsm_id_json = {row['id']: row for row in gsm_json}  # Associate ID to key value
        if gsm_id in gsm_id_json:
            return gsm_id_json[gsm_id]
        else:
            return None

    def get_node_id(self, node_id):
        value = self.node_id_map.get(node_id, node_id)
        if value is not None and value in self.node_id_map.keys():  # Check if value exists and is a key
            return self.node_id_map.get(value)
        else:
            return value

    def get_original_node_id(self, node_id, nodes):
        for old_id, new_id in self.node_id_map.items():
            # If we have MULTIINDIRECT, we just want the original ID as the passed on
            if old_id in nodes and nodes[old_id].type == Grouping.MULTIINDIRECT and len(
                    nodes[old_id].entities) == 1:
                return node_id

            if new_id == node_id:
                return old_id
        return node_id

    def resolve_node_id(self, node_id, nodes):
        if node_id is None:
            return Singleton(
                id=-1,
                named_entity="?" + str(self.existentials.increaseAndGetExistential()),
                properties=frozenset(),
                min=-1,
                max=-1,
                type='existential',
                confidence=1
            )
        else:
            return nodes[self.get_node_id(node_id)]

    def freshId(self):
        self.max_id += 1
        return self.max_id

    def freshIdAndAddToNodeMap(self, id_to_map):
        fresh_id = self.freshId()
        self.node_id_map[id_to_map] = fresh_id
        return fresh_id

    # Insert at position to retain topological order
    def add_to_nodes_at_pos(self, nodes, new_node, pos):
        items = list(nodes.items())
        items.insert(pos, (new_node.id, new_node))
        return dict(items)

    def is_node_but(self, gsm_item):
        return len(gsm_item['xi']) > 0 and 'but' in gsm_item['xi'][0].lower() and 'cc' in gsm_item['ell'][0].lower()

    def get_min_position(self, node):
        if isinstance(node, Singleton):
            node_props = dict(node.properties)
            if not 'pos' in node_props:
                return -1
            else:
                return int(float(node_props['pos']))
        else:
            return min(filter(lambda y: y > -1, map(lambda x: self.get_min_position(x), node.entities)))

    def node_bfs(self, edges, root_node_id):
        nodes = defaultdict(set)
        for x in edges:
            nodes[x.source.id].add(x.target.id)

        visited = set()
        q = deque()  # Create a queue for BFS

        # Mark the source node as visited and enqueue it
        visited.add(root_node_id)
        q.append(root_node_id)

        while q:
            id = q.popleft()
            for dst in nodes[id]:
                if dst not in visited:
                    visited.add(dst)
                    q.append(dst)

        return visited

    def get_node_type(self, node):
        return node.type if isinstance(node.type, str) else node.type.name

    def get_valid_nodes(self, nodes):
        valid_nodes = []
        for node in nodes:
            if node is not None:
                valid_nodes.append(node)

        if len(valid_nodes) > 0:
            return valid_nodes
        else:
            return -1