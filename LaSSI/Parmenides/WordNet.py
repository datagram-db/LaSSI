import functools
import json
from collections import defaultdict
from typing import Union, Iterable

import rdflib

from LaSSI.Parmenides.GSMObject import GSMObject

class RDFGsmObject(GSMObject):
    def __init__(self, idx, dv:'RDFGsmDatabase', type):
        self.dv = dv
        self.idx = rdflib.URIRef(idx) if isinstance(idx, str) else idx
        self.type = type

    @property
    def property_keys(self) -> Iterable[str]:
        return self.type["property"].keys()

    @property
    def containment_keys(self) -> Iterable[str]:
        return self.type["containment"].keys()

    @property
    def ell(self) -> list[str]:
        return self.type["ell"]

    @property
    def xi(self) -> list[str]:
        ls = []
        for type in self.type["ell"]:
            for object in self.dv.db.objects(self.idx, rdflib.URIRef(type)):
                if object.__class__ == rdflib.term.Literal:
                    ls.append(object.value)
                else:
                    ls.append(str(object))
        return ls

    def property(self, key: str) -> Union[str, float, int]:
        ls = []
        for predicate, _, _ in self.type["property"].get(key, []):
            for object in self.dv.db.objects(self.idx, rdflib.URIRef(predicate)):
                if object.__class__ == rdflib.term.Literal:
                    ls.append(object.value)
                else:
                    ls.append(str(object))
        assert len(ls)==1
        return ls[0]

    @property
    def scores(self) -> list[float]:
        return [1.0]

    def containment(self, key: str) -> list[tuple[float, 'GSMObject']]:
        ls = []
        for predicate, score_field, _ in self.type["containment"].get(key, []):
            for object in self.dv.db.objects(self.idx, rdflib.URIRef(predicate)):
                score = 1.0
                if score_field is not None:
                    ls = [x.value for x in self.dv.db.objects(self.idx, rdflib.URIRef(score_field))]
                    score = sum(ls)/len(ls)
                if object.__class__ == rdflib.term.Literal:
                    raise ValueError
                else:
                    ls.append([score, self.dv[object]])
        return ls

class RDFGsmDatabase:
    def __init__(self, db:rdflib.Graph, schema):
        self.db = db
        self.schema = schema

    def _getSchemaTypeOf(self, item):
        item = rdflib.URIRef(item) if isinstance(item, str) else item
        xs = [self.schema[str(x)] for x in self.db.objects(subject=item, predicate=rdflib.RDF.type, unique=True) if str(x) in self.schema]
        from LaSSI.Parmenides.schema import merge_schema_elements
        return functools.reduce(merge_schema_elements, xs, {"property":[], "ell":[], "xi":[], "containment":[]})

    def __getattr__(self, item):
        type = self._getSchemaTypeOf(item)
        return RDFGsmObject(item, self, type)


if __name__ == "__main__":
    file = "/media/gyankos/Biggus/conceptnet/data/raw/wordnet-rdf/wn31.nt"
    g = rdflib.Graph()
    g.parse(file)
    type_specification = defaultdict(lambda: defaultdict(set))
    for subject, type_info in g.subject_objects(predicate=rdflib.RDF.type):
        type_info = str(type_info)
        for predicate, object in  g.predicate_objects(subject):
            target_type = None
            type_of_field = None
            score_field = None
            if object.__class__ == rdflib.term.Literal:
                type_of_field = "property"
                target_type = object.datatype
            else:
                type_of_field = "containment"
                target_type = list(g.objects(object, predicate=rdflib.RDF.type))[0]
            target_type = str(target_type)
            if target_type == "http://www.w3.org/2000/01/rdf-schema#label":
                type_specification[type_info]["xi"].add(str(predicate))
            elif target_type == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                type_specification[type_info]["ell"].add(str(predicate))
            else:
                type_specification[type_info][type_of_field].add((str(predicate), score_field, str(target_type)))
    type_specification = {k: {k2:list(v2) for k2,v2 in v.items()} for k, v in type_specification.items()}
    with open("/data/nt_schema.json", "w") as f:
        json.dump(type_specification, f, indent=4)