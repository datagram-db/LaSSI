import ast
import glob
import json
import os.path
import pathlib

import dash
import yaml
from dash import dash_table
import dashvis
import flask
import dash_bootstrap_components as dbc
import pandas
from dash import Input, Output, dcc, html
from dashvis import DashNetwork

from LaSSI.structures.extended_fol.Enums import PairwiseCases
from LaSSI.viz import NestedTables
import dash_bootstrap_components as dbc

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
file = "/home/giacomo/projects/LaSSI/test_sentences/orig/cat_mouse.yaml"
from pathlib import Path
path = Path(file)
matrix_file = os.path.join(path.parent.absolute(), f"{path.stem}_matrix.json")

def generate_dropdown_menu(i, j):
    selected_value = "Equivalent" if i == j else "Indifferent"
    # dbc.Select(options=[
    #     {'label': 'eeney', 'value': 'eeney'},
    #     {'label': 'meeney', 'value': 'meeney'},
    #     {'label': 'miney', 'value': 'miney'},
    #     {'label': 'mo', 'value': 'mo'},
    # ], id='test-dropdown')
    return html.Select([html.Option(label=e.name,value=e.name,selected=e.name == selected_value) for e in PairwiseCases], id=f"cell-{i}-{j}", name=f"cell-{i}-{j}")


def generate_row(i, n):
    return html.Tr([html.Td(html.B(str(i)))]+[html.Td(generate_dropdown_menu(i,j+1)) for j in range(n)])

def generate_table(n):
    table_header = [html.Thead(html.Tr([html.Th(" - ")] + [html.Th(html.B(str(j+1))) for j in range(n)]))]
    table_body = [html.Tbody([generate_row(i+1, n) for i in range(n)])]
    return dbc.Table(table_header + table_body, bordered=True)

def isDebugMode():
    import sys
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    elif gettrace():
        return True
    else:
        return False

def build_form(sentences):
    n = len(sentences)
    # form = generate_table(n)

    form = html.Form([
        generate_table(n),
        html.Button('Submit', type='submit')
    ], action='/post', method='post')
    return html.Div([form, html.Ol([html.Li(s) for s in sentences])])

@app.server.route('/post', methods=['POST'])
def on_post():
    from flask import request
    global data_out
    data =  request.get_data()
    data = data.decode('utf8')
    N = -1
    d = dict()
    for x in data.split("&"):
        keys = x.split("=")
        coordinates = keys[0].split("-")
        coordinates = (int(coordinates[1]), int(coordinates[2]))
        N = max([coordinates[0],coordinates[1], N])
        val = PairwiseCases[keys[1]]
        d[coordinates] = val
    matrix = [[0.0 for _ in range(N)] for _ in range(N)]
    for (i,j) in d:
        val = d[(i,j)]
        if val == PairwiseCases.Implying or val == PairwiseCases.Equivalent:
            matrix[i - 1][j - 1] = 1.0
        elif val == PairwiseCases.ConflictingImplication:
            matrix[i - 1][j - 1] = 0.0
        else:
            matrix[i - 1][j - 1] = None
    with open(matrix_file, "w") as f:
        f.write(json.dumps(matrix, indent=4))
    return matrix


if __name__ == "__main__":

    with open(file, "r") as sentences_f:
        sentences = yaml.safe_load(sentences_f)
    app.layout = build_form(sentences)
    app.run(debug=isDebugMode())