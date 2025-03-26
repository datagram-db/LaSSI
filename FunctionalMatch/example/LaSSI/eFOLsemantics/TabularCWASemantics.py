import os
import pickle
import subprocess
from collections import defaultdict
from typing import List

import pandas
from functools import reduce

from FunctionalMatch.example.LaSSI.eFOLsemantics.ExpandConstituents import ExpandConstituents
from FunctionalMatch.example.parmenides.Formulae import Formula
from FunctionalMatch.example.parmenides.formula_utils import latex_rendering, latex_rendering_to_raster_file
from FunctionalMatch.utils import CountingDictionary

def png_node(obj, key, dir, nodes_map,fillColor=None):
    import pydot
    local_file = os.path.join(dir, key + ".svg")
    local_file_exless = os.path.join(dir, key )
    if not os.path.exists(local_file_exless):
        latex_rendering_to_raster_file(obj, local_file_exless)
    d = {"image": local_file,
     "label": "",
     "width": "3cm",
     "height": "1cm",
     "shape": "box"}
    if fillColor is not None:
        d["fillcolor"] = fillColor
        d["style"] = "filled"
    nodes_map[key] = pydot.Node(key, **d)
    return nodes_map

def with_variables_from(f, l, minimal_constituents: CountingDictionary, fn, selection=False):
    from FunctionalMatch.example.parmenides.formula_utils import semantic
    pdf = reduce(lambda x,y: x.merge(y, how="cross"),[pandas.DataFrame({str(x): [1,0]}) for x in l])
    L = []
    for x in pdf.to_dict(orient='records'):
        d = dict()
        for k,v in x.items():
            d[minimal_constituents.fromId(int(k))] = v
        x[fn] = semantic(f, d)
        if not selection or x[fn]>0.0:
            L.append(x)
    return pandas.DataFrame(L)

def with_true_variables_from(l):
    return pandas.DataFrame({str(x): [1] for x in l})


class TabularCWASemantics:
    def __init__(self, sentence_list:List[Formula], cache_folder):
        self.sentence_list = []
        self.sentence_to_id = dict()
        for idx in range(len(sentence_list)):
            self.sentence_to_id[sentence_list[idx]] = idx
            self.sentence_list.append(sentence_list[idx])
        # self.kb = kb
        self.minimal_constituents = CountingDictionary()
        self.U = None
        self.minimal_constituent_dict = defaultdict(set)
        self.buildup = False
        self.ec = None
        self.cache_folder = cache_folder

        #getSentenceAtomsFromId
        for sentence_id in range(len(self.sentence_list)):
            # collect_sentence_constituents
            from FunctionalMatch.example.parmenides.formula_utils import getAtoms
            # getSentenceAtomsFromId, for arg
            for x in getAtoms(self.sentence_list[sentence_id]):
                self.minimal_constituent_dict[sentence_id].add(self.minimal_constituents.add(x))

        d_folder = os.path.join(self.cache_folder, "_d.pickle")
        with open(d_folder, "wb") as p:
            pickle.dump(self.minimal_constituent_dict, p, protocol=pickle.HIGHEST_PROTOCOL)
        cd_folder = os.path.join(self.cache_folder, "_cd.pickle")
        with open(cd_folder, "wb") as p:
            pickle.dump(self.minimal_constituents, p, protocol=pickle.HIGHEST_PROTOCOL)
        self.ec = ExpandConstituents(self.cache_folder, self.minimal_constituents.getAllObjects())

    def getIDXGraph(self):
        return self.ec.getIDXGraph()

    def getConstituentFromIDX(self, idx):
        return self.ec.getConstituentFromIDX(idx)

    def getConstituentIDX(self, obj):
        return self.ec.getConstituentIDX(obj)

    def getImplExpansions(self, minimal_constituent_idx):
        return self.ec.getImplExpansions(minimal_constituent_idx)

    def getEqExpansions(self, minimal_constituent_idx):
        return self.ec.getEqExpansions(minimal_constituent_idx)

    def getImplExpansionExplanation(self, minimal_constituent_idx):
        return self.ec.getImplExpansionExplanation(minimal_constituent_idx)

    def getEqExpansionExplanation(self, minimal_constituent_idx):
        return self.ec.getEqExpansionExplanation(minimal_constituent_idx)

    def __call__(self, i, j):
        return self.get_straightforward_id_similarity(self.sentence_to_id[i], self.sentence_to_id[j])

    def _mutual_truth(self, i, j):
        test = self.ec.determine(i, j) #self.get_mutual_truth(i, j)
        # relation = Relation()
        # relation.add_attributes([str(i), str(j)])
        from FunctionalMatch.example.LaSSI.eFOLsemantics.Enums import PairwiseCases
        if (test == PairwiseCases.NonImplying):
            return pandas.DataFrame({str(i): [0,0,1,1],
                     str(j): [0,1,0,1]})
        elif (test == PairwiseCases.Implying):
            return pandas.DataFrame({str(i): [0, 0, 1],
                            str(j): [0, 1, 1]  })
        elif (test == PairwiseCases.MutuallyExclusive):
            return pandas.DataFrame({str(i): [0, 1],
                           str(j): [1,0]})
        elif (test == PairwiseCases.Equivalent):
            return pandas.DataFrame({str(i): [0, 1],
                           str(j): [0,1]})

    def _universal_truth(self, S, T):
        L = list()
        N = len(S.union(T))
        if N == 0:
            # relation = Relation(name="R")
            return pandas.DataFrame({})
        if N == len(S.intersection(T)) and N == 1:
            return pandas.DataFrame({str(list(S)[0]):[0,1]})
        else:
            for i in S:
                for j in T:
                    if i != j:
                        L.append(self._mutual_truth(i, j))
            return reduce(lambda x, y: x.merge(y), L)

    def get_straightforward_id_similarity(self, i, j):
        Ri = with_variables_from(self.sentence_list[i], self.minimal_constituent_dict[i], self.minimal_constituents, "R" + str(i), True)
        Rj = with_variables_from(self.sentence_list[j], self.minimal_constituent_dict[j], self.minimal_constituents, "R" + str(j))
        result = self._universal_truth(set(self.minimal_constituent_dict[i]), set(self.minimal_constituent_dict[j])).merge(Ri).merge(Rj)[list(set(Ri.columns).union(set(Rj.columns)))].drop_duplicates()[["R" + str(j)]].prod(axis=1)
        total = result.sum(axis=0)/len(result)
        # print(f"{i}~{j} := {total}")
        return total

    def buildReport(self, file, mathJax = True):
        from bs4 import Tag, BeautifulSoup
        import pydot
        from FunctionalMatch.example.parmenides.formula_utils import latex_formula_rendering

        from pathlib import Path
        Path(file+"_dir").mkdir(parents=True, exist_ok=True)
        graph = pydot.Dot("my_graph", graph_type="digraph", rankdir="LR")
        nodes_map = dict()
        html = Tag(name="html")
        if mathJax:
            mathjax = """
            <script type="text/javascript" id="MathJax-script" async
      src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.0.0/es5/latest?tex-mml-chtml.js">
    </script>
    <script>
    MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']]
      }
    };
    </script>
    <script id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
    </script><script src="//d3js.org/d3.v7.min.js"></script>
<script src="https://unpkg.com/@hpcc-js/wasm@2.20.0/dist/graphviz.umd.js"></script>
<script src="https://unpkg.com/d3-graphviz@5.6.0/build/d3-graphviz.js"></script>"""
            parser = BeautifulSoup(mathjax)
            for x in list(parser.children):
                html.append(x)
        body = Tag(name="body")

        p_ = Tag(name="h1")
        p_.append("Constutents DB")
        body.append(p_)
        ol = Tag(name="ol")
        initial_constituents = self.minimal_constituents.getAllObjects()
        # constituent_id = CountingDictionary()

        for idx_orig, x in enumerate(initial_constituents):
            li = Tag(name="li")
            idx = self.getConstituentIDX(x)
            li["id"] = f"constituent{idx}"
            li.append(latex_formula_rendering(x, mathJax))

            pp = Tag(name="p")
            pp.append("Eq Rewriting:")
            li.append(pp)
            ool = Tag(name="ol")
            # eqex = self.getEqExpansions(idx)
            # meqex = {x:idx2+1 for idx2, x in enumerate(eqex)}
            fmeqex = self.getEqExpansionExplanation(idx_orig)
            for idx_ in fmeqex:
                lli = Tag(name="li")
                lli["value"] = idx_
                x = self.ec.getIthExpandedConstituent(idx_)
                lli.append(latex_formula_rendering(x, mathJax))
                uuul = Tag(name="ul")
                for (label, rule),dst in fmeqex[idx_]:
                    assert label == "eqR"
                    llli = Tag(name="li")
                    llli.append(str(idx_)+latex_rendering(f"\\xrightarrow{{eq {rule}}}")+str(dst))
                    uuul.append(llli)
                lli.append(uuul)
                ool.append(lli)
            li.append(ool)

            pp = Tag(name="p")
            pp.append("Impl Rewriting:")
            li.append(pp)
            # imex = self.getImplExpansions(idx)
            # mimex = {x:idx2+1 for idx2, x in enumerate(imex)}
            fmimex = self.getImplExpansionExplanation(idx_orig)
            ool = Tag(name="ol")
            for idx_ in fmimex:
                lli = Tag(name="li")
                lli["value"] = idx_
                x = self.ec.getIthExpandedConstituent(idx_)
                lli.append(latex_formula_rendering(x, mathJax))
                uuul = Tag(name="ul")
                for (label, rule), dst in fmimex[idx_]:
                    assert label == "implR"
                    llli = Tag(name="li")
                    llli.append(str(idx_)+latex_rendering(f"\\xrightarrow{{impl {rule}}}")+str(dst))
                    uuul.append(llli)
                lli.append(uuul)
                ool.append(lli)
            li.append(ool)

            ol.append(li)
        body.append(ol)
        print("Finished to write the rules")

        p_ = Tag(name="h1")
        p_.append("Sentences DB")
        body.append(p_)
        uul = Tag(name="ul")
        body.append(uul)

        for i, sentence in enumerate(self.sentence_list):
            minimal_constituents = self.minimal_constituent_dict[i]
            ref = f"Sentence{i}"
            Sentence = f"Sentence #{i}"
            print(ref)
            nodes_map[ref] = pydot.Node(ref, shape="circle",fillcolor="lightyellow",style="filled")
            graph.add_node(nodes_map[ref])


            lli = Tag(name="li")
            lla = Tag(name="a")
            lla["href"] = f"#{ref}"
            lla.append(Sentence)
            lli.append(lla)
            uul.append(lli)

            h1 = Tag(name="h2")
            a = Tag(name="a")
            h1["id"] = ref
            a.append(Sentence)
            h1.append(a)
            body.append(h1)

            p1 = Tag(name="p")
            p1.append("Logic form: ")
            p1.append(latex_formula_rendering(sentence, mathJax))
            body.append(p1)
            nodes_map = png_node(sentence, ref+"eq", file+"_dir", nodes_map)
            graph.add_node(nodes_map[ref+"eq"])
            graph.add_edge(pydot.Edge(ref, ref+"eq", label="hasFormula"))

            p2 = Tag(name="p")
            p2.append("Minimal Constituents:")
            body.append(p2)
            ol = Tag(name="ul")
            for x_idx in minimal_constituents:
                obj = initial_constituents[x_idx]
                global_x_idx = self.getConstituentIDX(obj)
                li = Tag(name="li")
                li["value"] = global_x_idx
                ali = Tag(name="a")
                ali["href"] = f"#constituent{global_x_idx}"
                ali.append(str(global_x_idx))
                ali.append(latex_formula_rendering(obj, mathJax))
                nodes_map = png_node(initial_constituents[x_idx], f"constituent{global_x_idx}", file + "_dir", nodes_map, fillColor="lightblue")
                graph.add_node(nodes_map[f"constituent{global_x_idx}"])
                graph.add_edge(pydot.Edge(ref, f"constituent{global_x_idx}", label="hasConstituent"))
                li.append(ali)
                ol.append(li)
            body.append(ol)
        html.append(body)
        html.decode()

        print("Printing html...")
        with open(file+".html", "w") as f:
            f.write(html.prettify())
        print("... done")

        graphIDX = self.getIDXGraph()
        for idx, ls in graphIDX.items():
            print(f"Node {idx}")
            label = f"constituent{idx}"
            if label not in nodes_map:
                src_obj = self.getConstituentFromIDX(idx)
                nodes_map = png_node(src_obj, label, file + "_dir", nodes_map)
                graph.add_node(nodes_map[label])
            # else:
            #     src_obj = nodes_map[label]
            for (ruleId, dstIdx) in ls:
                labelDst = f"constituent{dstIdx}"
                if labelDst not in nodes_map:
                    dst_obj = self.getConstituentFromIDX(dstIdx)
                    nodes_map = png_node(dst_obj, labelDst, file + "_dir", nodes_map)
                    graph.add_node(nodes_map[labelDst])
                # else:
                #     dst_obj = nodes_map[labelDst]
                graph.add_edge(pydot.Edge(label, labelDst, label=str(ruleId)))
        print("Graph finalised")

        with open(file+".dot", "w") as f:
            # As a string:
            output_raw_dot = graph.to_string()
            from FunctionalMatch.example.utils.doc2tex import convert_graph
            # tex_graph = convert_graph(output_raw_dot)
            f.write(output_raw_dot)
        print("Dot written")

        graph.write(file + ".pdf", format="pdf")
        print("PDF written")
        # with open(file+".dot", "w") as f:
        #     # As a string:
        #     output_raw_dot = graph.to_string()
        #     from FunctionalMatch.example.utils.doc2tex import convert_graph
        #     # tex_graph = convert_graph(output_raw_dot)
        #     f.write(output_raw_dot)
        #     # Or, save it as a DOT-file:
        #     # import pdflatex
        #     # bin_graph = str.encode(tex_graph)
        #     # pdfl = pdflatex.PDFLaTeX.from_binarystring(bin_graph, file)
        #     # pdfl.params["-output-directory"] = os.getcwd()
        #     # pdfl.create_pdf(keep_pdf_file=True)
        #     # fp = subprocess.run(["pdflatex", file+".tex"])


