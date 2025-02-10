"""
This app creates a simple sidebar layout using inline style arguments and the
dbc.Nav component.

dcc.Location is used to track the current location, and a callback uses the
current location to render the appropriate page content. The active prop of
each NavLink is set automatically according to the current pathname. To use
this feature you must install dash-bootstrap-components >= 0.11.0.

For more details on building multi-page Dash applications, check out the Dash
documentation: https://dash.plot.ly/urls
"""
import ast
import glob
import json
import os.path
import pathlib

import dash
import dash_table
import dashvis
import flask
import dash_bootstrap_components as dbc
import pandas
from dash import Input, Output, dcc, html
from dashvis import DashNetwork

from LaSSI.viz import NestedTables


def convert_html_to_dash(html_code, dash_modules=None):
    """Convert standard html (as string) to Dash components.

    Looks into the list of dash_modules to find the right component (default to [html, dcc, dbc])."""
    from xml.etree import ElementTree

    if dash_modules is None:
        import dash_html_components as html
        import dash_core_components as dcc

        dash_modules = [html, dcc]
        try:
            import dash_bootstrap_components as dbc

            dash_modules.append(dbc)
        except ImportError:
            pass

    def find_component(name):
        for module in dash_modules:
            try:
                return getattr(module, name)
            except AttributeError:
                pass
        raise AttributeError(f"Could not find a dash widget for '{name}'")

    def parse_css(css):
        """Convert a style in ccs format to dictionary accepted by Dash"""
        return {k: v for style in css.strip(";").split(";") for k, v in [style.split(":")]}

    def parse_value(v):
        try:
            return ast.literal_eval(v)
        except (SyntaxError, ValueError):
            return v

    parsers = {"style": parse_css, "id": lambda x: x}

    def _convert(elem):
        comp = find_component(elem.tag.capitalize())
        children = [_convert(child) for child in elem]
        if not children:
            children = elem.text
        attribs = elem.attrib.copy()
        if "class" in attribs:
            attribs["className"] = attribs.pop("class")
        attribs = {k: parsers.get(k, parse_value)(v) for k, v in attribs.items()}

        return comp(children=children, **attribs)

    et = ElementTree.fromstring(html_code)

    return _convert(et)

# app = dash.Dash()
meuDB = []

def create_dash_app(requests_pathname_prefix: str = None):
    """
    Sample Dash application from Plotly: https://github.com/plotly/dash-hello-world/blob/master/app.py
    """

    server = flask.Flask(__name__)
    full_path = os.path.join(os.getcwd(), "catabolites", requests_pathname_prefix)
    app = dash.Dash(__name__, server=server, requests_pathname_prefix=f"/{requests_pathname_prefix}/", external_stylesheets=[dbc.themes.BOOTSTRAP,dashvis.stylesheets.VIS_NETWORK_STYLESHEET])
    if not os.path.exists(full_path) or not os.path.isdir(full_path):
        return app

    lines = []
    with open(os.path.join(full_path, "string_rep.txt"), "r") as f:
        for line in f:
            actual_line = line.split(" ⇒ ")[0]
            lines.append(actual_line)


    # the style arguments for the sidebar. We use position:fixed and a fixed width
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    }

    # the styles for the main content position it to the right of the sidebar and
    # add some padding.
    CONTENT_STYLE = {
        "margin-left": "18rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    }

    navlinks = [dbc.NavLink("Home", href="/", active="exact")]
    for idx, sentence in enumerate(lines):
        navlinks.append(dbc.NavItem(dbc.NavLink(f"{idx}. {sentence}", href=f"/{idx}", active="exact")))

    LS = []
    with open(os.path.join(full_path, "logical_rewriting.json"), "r") as f:
        data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == len(lines)
        for logical in data:
            from LaSSI.structures.extended_fol.Sentences import formula_from_dict
            actual = formula_from_dict(logical)

            import latextools

            # Render latex
            latex_eq = latextools.render_snippet(
                r'$'+str(actual)+'$',
                commands=[latextools.cmd.all_math])

            svg_eq = latex_eq.as_svg()
            import base64
            encoded = base64.b64encode(str.encode(svg_eq.content))
            svg = 'data:image/svg+xml;base64,{}'.format(encoded.decode())
            LS.append(html.Img(src=svg))

    matrices = dict()
    ls = glob.glob(f'{full_path}/confusion_matrices_*.json')
    for x in ls:
        type = pathlib.Path(x).name[19:-5]
        z = []
        with open(x, "r") as f:
            val = f.read()
            print(val)
            z = json.loads(val)
        import plotly.figure_factory as ff

        x = y = [str(len + 1) for len in range(len(lines))]
        z_text = [[str(round(y,2)) for y in x] for x in z]

        import plotly.graph_objects as go

        fig = go.Figure(data=go.Heatmap(
            z=z,
            text=z_text,
            texttemplate="%{text}",
            textfont={"size": 20}, colorscale='Viridis'))

        # fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
        # fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
        #                   # xaxis = dict(title='x'),
        #                   # yaxis = dict(title='x')
        #                   )
        # # adjust margins to make room for yaxis title
        # fig.update_layout(margin=dict(t=50, l=200))
        #
        # # add colorbar
        # fig['data'][0]['showscale'] = True
        matrices[type] = fig


    with open(os.path.join(full_path, "meuDBs.json"), "r") as f:
        for x in json.load(f):
            meuDB.append(sorted(x["multi_entity_unit"], key=lambda x: x["confidence"], reverse=True))

    sidebar = html.Div(
        [
            html.H2("Sentences", className="display-4"),
            html.Hr(),
            html.P(
                "The list of the sentences from the database", className="lead"
            ),
            dbc.Nav(
                navlinks,
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
    )

    content = html.Div(id="page-content", style=CONTENT_STYLE)
    app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def render_page_content(pathname):
        try:
            val = int(pathname[1:])
            df = pandas.DataFrame(meuDB[val])
            from LaSSI.viz import NestedTables
            table = NestedTables.generate_morphism_html(os.path.join(full_path, "viz"),
                                                pathname[1:])
            return html.Div([html.H1("Original sentence"),
                      html.Hr(),
                      html.P(lines[val]),
                      html.H1("Logical Representation"),
                      html.Hr(),
                      LS[val],
                             html.H1("Morphisms"),
                             table,
                        html.H1("MeuDB"),
                        dash_table.DataTable(meuDB[val],  [{"name": i, "id": i} for i in df.columns])])
        except:
            pass
        if pathname == "/":
            matrices_final = []
            for matrix_name, fig in matrices.items():
                matrices_final.append(html.H2(matrix_name))
                matrices_final.append(dcc.Graph(figure=fig))
            if len(matrices_final) > 0:
                matrices_final.insert(0, html.H1("Confusion Matrices"))
                return html.Div(matrices_final)
        # If the user tries to reach a different page, return a 404 message
        return html.Div(
            [
                html.P(f"Please select an item from the right bar"),
            ],
            className="p-3 bg-light rounded-3",
        )
    return app

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware


app = FastAPI()
catabolites = []
desktop = pathlib.Path(os.path.join(os.getcwd(), "catabolites"))
for item in desktop.iterdir():
    if item.is_dir():
        catabolites.append(str(item.name))
# catabolites = [x[0] for x in os.walk(os.path.join(os.getcwd(), "catabolites"))]

@app.get("/")
def read_main():
    routes = [{"method": "GET", "path": "/", "summary": "Landing"},{"method": "GET", "path": "/status", "summary": "App status"}]
    for x in catabolites:
        routes.append({"method": "GET", "path": f"/{x}", "summary": f"Sub-mounted Dash app for {x}"})
    return {
        "routes": routes
    }

@app.get("/status")
def get_status():
    return {"status": "ok"}

for x in catabolites:
    dash_app = create_dash_app(requests_pathname_prefix=x)
    app.mount(f"/{x}", WSGIMiddleware(dash_app.server))

if __name__ == "__main__":
    # htmltbl = NestedTables.generate_morphism_html(os.path.join("/home/giacomo/projects/LaSSI/catabolites/alice_bob", "viz"),
    #                                     "0")
    # # table = convert_html_to_dash(htmltbl)
    # print(table)
    uvicorn.run(app, port=8000)

