from igraph import *

class IGraph:
    def __init__(self, obj):
        self.g = Graph()
        self.node_properties = set()
        self.edge_properties = set()
        obj = list(obj)
        obj.sort(key=lambda x: int(x['id']))
        nodes = set()
        for node in obj:
            idx = str(node['id'])
            nodes.add(idx)
            n = self.g.add_vertex(idx)
            n["_ell"] = node.get("ell", list())
            n["_xi"] = node.get("xi", list())
            if "properties" in node:
                for k,v in node["properties"].items():
                    self.node_properties.add(k)
                    n[k] = v
        for node in obj:
            if "phi" in node:
                for edge in node["phi"]:
                    if "parent" in edge and "child" in edge:
                        parent = str(edge["parent"])
                        child = str(edge["child"])
                        if parent in nodes and child in nodes:
                            label = edge.get("containment", "")
                            edge = self.g.add_edge(parent, child)
                            if "properties" in edge:
                                for k, v in edge["properties"].items():
                                    self.edge_properties.add(k)
                                    edge[k] = v
                            edge["_label"] = label

    def vertexSetSize(self):
        return len(self.g.vs)

    def getNodeLabels(self, idx):
        if len(self.g.vs) > idx:
            return self.g.vs[idx]["_ell"]
        else:
            return None

    def getNodeValues(self, idx):
        if len(self.g.vs) > idx:
            return self.g.vs[idx]["_xi"]
        else:
            return None

    def hasNodeProperty(self, idx, prop):
        if len(self.g.vs) > idx:
            return prop in self.g.vs[idx]
        else:
            return False

    def getNodeProperty(self, idx, prop):
        if len(self.g.vs) > idx:
            return self.g.vs[idx][prop]
        else:
            return None

    @staticmethod
    def from_json_file(file):
        ls_graph = []
        with open(file) as f:
            import json
            ls_graph = json.load(f)
        return [IGraph(obj) for obj in ls_graph]

if __name__ == "__main__":
    file = "/home/giacomo/projects/LaSSI/catabolites/alice_bob/datagramdb_output.json"
    ls = IGraph.from_json_file(file)
    print(ls)