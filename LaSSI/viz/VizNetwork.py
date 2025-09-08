import json
import os

import ast

height_ = '100%'
width_ = '100%'

nodes_ = {
    'shape': 'box',
    'margin': 10,
    'size': 25,
    'borderWidth': 2,
    'borderWidthSelected': 2,
    'font': {
        'multi': 'markdown',
        'align': 'center',
    },
    'labelHighlightBold': True,
    'widthConstraint': {
        'minimum': 30,
        'maximum': 100,
    }
}

edges_ = {
    'color': {
        'inherit': 'both',
    },
    'arrows': {
        'to': {
            'enabled': True,
            'scaleFactor': 0.5
        }
    },
    'chosen': False,
    "arrowStrikethrough": False,
    'smooth': {
        'type': "dynamic",
        'roundness': 0.5,
    }
}

physics_ = {
    'enabled': True,
}

manipulation_ = {
    'enabled': True,
    'initiallyActive': True,
    'addNode': """function(nodeData,callback) {
      nodeData.label = 'hello world';
      callback(nodeData);
}""",
    'addEdge': True,
    'editNode': None,
    'editEdge': True,
    'deleteNode': True,
    'deleteEdge': True,
}

interaction_ = {
    'hover': True,
    'hoverConnectedEdges': True,
    'multiselect': True,
    'keyboard': {
        'enabled': True,
        'bindToWindow': False,
        'autoFocus': True,
    },
    'navigationButtons': True,
}

layout_ = {
    'improvedLayout': True,
}

configure_ = {
    'enabled': False,
    'showButton': False,
}

default_options_ = dict(autoResize=True, height=height_, width=width_, configure=configure_, nodes=nodes_, edges=edges_,
                        layout=layout_, interaction=interaction_, manipulation=manipulation_, physics=physics_, )


def load_nodes_any(folder, N, no, file):
    if folder not in N:
        N[no] = dict()
        from LaSSI.viz.ReadGSMExt import deserialize_gsm_file
        for obj in deserialize_gsm_file(os.path.join(os.getcwd(), "catabolites", folder, "viz", str(no), file)):
            N[no][obj.id] = obj
    return N


def load_edges_any(folder, N, E, no, file):
    N = load_nodes_any(folder, N,no, file)
    if folder not in E:
        from LaSSI.viz.GSMExt import to_vis_network_phi
        E[no] = to_vis_network_phi(N[no].values())
    return (N, E)


class VizNetwork(object):
    def __init__(self):
        self.N_result = dict()
        self.E_result = dict()
        self.N_input = dict()
        self.E_input = dict()
        self.N_removed = dict()
        self.N_inserted = dict()

    def result_nodes(self,folder, no):
        self._load_input_nodes(folder, no)
        self._load_result_nodes(folder, no)
        from LaSSI.viz.GSMExt import to_vis_nodes
        return to_vis_nodes(self.N_result[no].values(), None, self.N_inserted[no])

    def input_nodes(self,folder,no):
        self._load_input_nodes(folder,no)
        from LaSSI.viz.GSMExt import to_vis_nodes
        return to_vis_nodes(self.N_input[no].values(), self.N_removed[no], self.N_inserted[no])

    def edges(self,folder,no):
        self._load_result_edges(folder,no)
        return self.E_result[no]

    def input_edges(self,folder,no):
        self._load_input_edges(folder,no)
        return self.E_input[no]

    def _load_result_nodes(self,folder,no):
        self.N_result = load_nodes_any(folder, self.N_result, no,"result.json")

    def _load_input_nodes(self,folder,no):
        with open(os.path.join(os.getcwd(), "catabolites", folder, "viz", no, "removed.json"), "r") as f:
            self.N_removed[no] = set(json.load(f))
        with open(os.path.join(os.getcwd(), "catabolites", folder, "viz",no, "inserted.json"), "r") as f:
            self.N_inserted[no] = set(json.load(f))
        self.N_input = load_nodes_any(folder, self.N_input, no, "input.json")

    def _load_result_edges(self,folder, no):
        (self.N_result, self.E_result) = load_edges_any(folder, self.N_result, self.E_result, no,"result.json")

    def _load_input_edges(self,folder,no):
        (self.N_input, self.E_input) = load_edges_any(folder, self.N_input, self.E_input, no,"input.json")

    def as_input_network(self, folder, no):
        from dashvis import DashNetwork
        network = DashNetwork(
            id='network',
            style={'height': '400px'},
            options=default_options_,
            enableHciEvents=False,
            enablePhysicsEvents=False,
            enableOtherEvents=False
        )
        data = {'nodes': self.input_nodes(folder,no), 'edges': self.input_edges(folder,no)}