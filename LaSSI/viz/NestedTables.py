__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2023"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "3.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

import os
import re
import glob

import pandas
from dash import html


def toSerialize(cp):
    if cp[1] == "graph":
        return str(cp[0])
    else:
        if isinstance(cp[0], int) or isinstance(cp[0], float):
            return "<a href=\"#\">"+str(cp[0])+"</a>"
        else:
            return str(cp[0])


def merge_tables(ll):
    if len(ll) == 0:
        return ll
    elif len(ll) == 1:
        return ll[0]
    elif len(ll) >= 2:
        import itertools
        list_1 = ll[0]
        list_2 = ll[1]
        from itertools import product
        ret_list = []
        for i1, i2 in product(list_1, list_2):
            merged = {}
            merged.update(i1)
            merged.update(i2)
            ret_list.append(merged)
        return merge_tables([ret_list] + ll[2:])

class Table:
    def __init__(self, Schema):
        self.Schema = Schema
        self.rows = []

    def addRow(self, row):
        if row is not None:
            self.rows.append(row)

    def as_list_dict(self, obj=()):
        L = []
        dd = dict()
        S = set()
        hasNested = False
        for row in self.rows:
            original = {}
            for k,v in zip(self.Schema, row):
                if v is not None:
                    if type(v).__name__ == "Table":
                        hasNested = True
                        if k in S:
                            S.remove(k)
                        rec, _ = v.as_list_dict(obj+(k,))
                        dd[k] = rec
                    else:
                        S.add(k)
                        original[obj+(k,)] = v
            LL = list(dd.values())
            LL.append([original])
            L += merge_tables(LL)
        return L, [(x,) for x in S] if hasNested else []

    def header(self):
        return "<thead><tr>"+("\t".join(map(lambda x: "<th>"+x+"</th>", self.Schema)))+"</tr></thead>"

    def body(self):
        return "<tbody>"+("".join(map(lambda x: "<tr >"+("".join(map(lambda y: "<td>"+toSerialize(y)+"</td>", zip(x,self.Schema))))+"</tr>", self.rows)))+"</tbody>"

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        # ls = map(lambda x: map(lambda y: str(y), x), self.rows)
        # return str(tabulate.tabulate(ls, self.Schema, tablefmt="html"))
        return "<table class=\"pure-table\">"+self.header()+self.body()+"</table>"


def parseDoubleQuote(s:str, quote:str):
    if s is None:
        return None
    s = s.strip()
    firstOffset = s.find(quote)
    if firstOffset!=0:
        return None
    s = s[firstOffset+len(quote):]
    firstOffset = s.find(quote)
    if firstOffset == -1:
        return None
    return (s[:firstOffset], s[firstOffset+len(quote):].strip())

def skipFirst(s:str, val:str):
    if s is None:
        return None
    s = s.strip()
    if not s.startswith(val):
        return None
    else:
        return s[len(val):]

def parse_cell(s:str, G):
    if s is None:
        return None
    s = s.strip()
    m = re.search(r"^[-+]?(?:\b[0-9]+(?:\.[0-9]*)?|\.[0-9]+\b)(?:[eE][-+]?[0-9]+\b)?", s)
    if s.startswith("@"):
        return parse_nested_table(s[1:], G, G)
    if s.startswith("NULL"):
        return (None,s[4:].strip())
    elif s.startswith('"'):
        return parseDoubleQuote(s, '"')
    elif m:
        val = s[m.start(0):m.end(0)]
        try:
            return (int(val), s[m.end(0):].strip())
        except:
            return (float(val), s[m.end(0):].strip())
        return None
    return None


def parse_row(schema:list[str], s:str, G:int):
    row = list()
    s = skipFirst(s, "[")
    if s is None:
        return None
    for x in range(len(schema)):
        val = parse_cell(s, G)
        if val is None:
            return None
        row.append(val[0])
        s = skipFirst(val[1], ',')
        if s is None:
            s = skipFirst(val[1], ']')
            if s is None:
                return None
            else:
                return (row, s)
    return None


def parse_nested_table(s:str, G, currentGraph = None):
    G = int(G)
    schema = list()
    # table = list()
    # s = skipFirst(s, ',')
    firstOffset = s.find('(')
    if firstOffset == -1:
        return None
    name = s[:firstOffset]
    s = s[firstOffset+1:]
    tmp = parseDoubleQuote(s, '"')
    rest = None
    while tmp is not None:
        schema.append(tmp[0])
        s = skipFirst(tmp[1], ',')
        if s is None:
            s = tmp[1]
            break
        tmp = parseDoubleQuote(s, '"')
    # df = pd.DataFrame(columns=schema)
    s = skipFirst(skipFirst(s, ')'), '{')
    table = Table(schema)
    g = -1
    if 'graph' in schema:
        g = schema.index('graph')
    while s is not None:
        result = parse_row(schema, s, G)
        if result is not None and (((g==-1)) or (result[0][g] <= G)):
            if (result[0][g] == G) or (currentGraph is not None):
                table.addRow(result[0])
            s = result[1]
        else:
            rest = skipFirst(s, "}")
            break

    return (table, rest)

# pure_min = """
# """

def generate_morphism_html(path, name):
    cwd = os.getcwd()
    os.chdir(path)
    d = dict()
    for file in glob.glob("*.ncsv"):
        toParse = file[:-5]
        lastPar = toParse.rfind(')')
        firstPar = toParse.rfind('(')
        no = int(toParse[firstPar + 1:lastPar])
        morph = toParse[:firstPar]
        if morph not in d:
            d[morph] = dict()
        with open(os.path.join(path, file), "r") as f:
            s = f.read()
            df = parse_nested_table(s, name)
            if df is not None:
                L, S = df[0].as_list_dict()
                if len(L)>0:
                    import dash_bootstrap_components as dbc
                    df = pandas.DataFrame.from_records(L)
                    if (len(S)>0):
                        df.set_index(S, inplace=True)
                    d[morph][no] = dbc.Table.from_dataframe(
                        df, striped=True, bordered=True, hover=True, index=True
                    )
        # print(file)

    L = []
    for morph, phases in d.items():
        if len(phases)>0:
            L.append(html.H3(morph))
            for x in phases.values():
                L.append(x)
    return html.Div(L)
    # strR = ""
    # for x in d:
    #     d[x] = "".join(map(lambda x: "" + str(x[1]) + "", sorted(d[x].items(), key=lambda x: int(x[0]))))
    # for header in sorted(d.keys()):
    #     strR = strR + "<h3>" + header + "</h3>" + d[header]
    # # element = """
    # #
    # # """
    # # strR = strR + "</body></html>"
    # # with open("morphisms.html", "w") as f:
    # #     f.write(strR)
    # # with open("pure-min.css", "w") as f:
    # #     f.write(pure_min)
    # os.chdir(cwd)
    # return "<div>"+strR+"</div>"


if __name__ == "__main__":
    htmltbl = generate_morphism_html(os.path.join("/home/giacomo/projects/LaSSI/catabolites/alice_bob", "viz"),
                                        "0")
    # table = convert_html_to_dash(htmltbl)
    print(htmltbl)