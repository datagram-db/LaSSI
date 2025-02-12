import csv
import dataclasses
import json
import os
from typing import List

textrepr = dict()
textlang = dict()
used_rel = dict()



class Relation:
    relation_id:str
    start_id: str
    end_id: str
    weight:str
    rel:str
    surfaceStart:str
    surfaceEnd:str
    langStart:str
    langEnd:str
    startRS_surface:str
    endRS_surface:str

    def __init__(self, line:List[str], headers:List[str]=None):
        if headers is None:
            headers = ["id", "uri", "relation_id", "start_id", "end_id", "weight", "data"]
        d = dict(zip(headers, line))
        try:
            d["data"] = json.loads(d["data"])
        except:
            d["data"] = json.loads(d["data"].replace('\\\\"', '\\"'))
        self.relation_id = d["relation_id"]
        self.start_id = d["start_id"]
        self.end_id = d["end_id"]
        self.weight = d["weight"]
        self.raw_rel = d["data"]["rel"]
        self.rel = self.raw_rel.replace("/r/", "").replace("dbpedia/", "")
        self.surfaceStart = d["data"]["surfaceStart"]
        self.surfaceEnd = d["data"]["surfaceEnd"]
        self.start = d["data"]["start"]
        self.end = d["data"]["end"]
        self.startR = self.start.replace("_", " ")
        self.startRS = self.startR.split('/')
        self.langStart = self.startRS[2]
        self.endR = self.end.replace("_", " ")
        self.endRS = self.endR.split('/')
        self.langEnd = self.endRS[2]
        self.startRS_surface = self.startRS[3]
        self.endRS_surface = self.startRS[3]
        if self.surfaceStart is None:
            self.surfaceStart = self.startRS[3]
        if self.surfaceEnd is None:
            self.surfaceEnd = self.endRS[3]

def non_word_elements(files, words:set, headers=None, lang=None):
    maxVertId = 0
    if headers is None:
        headers=["id", "uri", "relation_id", "start_id", "end_id", "weight", "data"]
    with open(files) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):
            r = Relation(line, headers)
            if lang is not None and ((r.langStart != lang) or (r.langEnd != lang)):
                continue
            if r.startRS_surface.lower() not in words and r.surfaceStart.lower() not in words:
                    yield r.surfaceStart
            if r.endRS_surface.lower() not in words and r.surfaceEnd.lower() not in words:
                    yield r.surfaceEnd


def declunk_edge_file(files, headers=None, lang=None):
    if headers is None:
        headers=["id", "uri", "relation_id", "start_id", "end_id", "weight", "data"]
    num_lines = 0
    i = 0

    maxVertId = 0
    with open(files) as tsv:
         with open(os.path.join(os.path.dirname(files), "conceptnet5_rectified.tab"), "w") as result_file:
            wr = csv.writer(result_file, dialect="excel-tab")

            wr.writerow(["surfaceStart", "start_id", "relation_id", "weight", "surfaceEnd", "end_id"])

            for line in csv.reader(tsv, dialect="excel-tab"): #You can also use delimiter="\t" rather than giving a dialect.
                r = Relation(line)

                if lang is not None and ((r.langStart != lang) or (r.langEnd != lang)):
                    continue
                if r.relation_id not in used_rel:
                    used_rel[r.relation_id] = r.raw_rel

                if not r.start_id in textrepr:
                    textrepr[r.start_id] = set()
                if not r.end_id in textrepr:
                    textrepr[r.end_id] = set()
                textrepr[r.start_id].add(r.surfaceStart.strip())
                textrepr[r.end_id].add(r.surfaceEnd.strip())
                textlang[r.start_id] = r.langStart
                textlang[r.end_id] = r.langEnd

                wr.writerow([r.surfaceStart, r.start_id, r.rel, r.weight, r.surfaceEnd, r.end_id])
                i = i+1

                maxVertId = max(maxVertId, int(r.start_id))

def numberbatch_parsing(file):
    import pandas
    f = pandas.read_hdf(file, 'mat', encoding='utf-8')
    for x in f.index:
        concept = x.split("/")
        if concept[2] == "en":
            key = concept[3].replace("_", " ")
            yield [key]


def get_triplets(file, lang=None):
    with open(file) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):
            r = Relation(line)

            if lang is not None and ((r.langStart != lang) or (r.langEnd != lang)): continue

            yield (r.surfaceStart, r.rel, r.surfaceEnd)



# if __name__ == '__main__':
#     file = '/media/giacomo/Biggus/conceptnet/data/psql/edges.csv'
#     declunk_edge_file(file, ["id", "uri", "relation_id", "start_id", "end_id", "weight", "data"], "en")