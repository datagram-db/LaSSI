__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2020, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

from collections import OrderedDict

import psycopg2
from psycopg2 import Error

from LaSSI.files.ReadFileContent import ReadFileContent


class FuzzyStringMatchDatabase:
    _instance = None

    def create(self, tablename, file):
        exists = False
        with self.connection.cursor() as cursor:
            cursor.execute("select * from information_schema.tables")
            records = cursor.fetchall()
            S = set(map(lambda x: x[2], records))
            exists = tablename in S
            cursor.close()
        if not exists:
            print("Creating table " + tablename)
            with self.connection.cursor() as cursor2:
                cursor2 = self.connection.cursor()
                cursor2.execute("DROP TABLE IF EXISTS " + tablename)
                self.connection.commit()
                cursor2.execute("CREATE TABLE " + tablename + " (id integer NOT NULL, idx text, t text)")
                # self.connection.commit()
                cursor2.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
                # self.connection.commit()
                from LaSSI.files.ReadFileContent import ReadFileContent
                with ReadFileContent(file) as f:
                    # Notice that we don't need the csv module.
                    next(f)  # Skip the header row.
                    cursor2.copy_from(f, tablename, sep='\t')
                cursor2.execute("CREATE INDEX " + tablename + "_idx ON " + tablename + " USING GIST (t gist_trgm_ops);")
                self.connection.commit()
                cursor2.close()
        else:
            print(f"Table {tablename} already loaded!")

    def create_typed_table(self, tablename, file):
        exists = False
        with self.connection.cursor() as cursor:
            cursor.execute("select * from information_schema.tables")
            records = cursor.fetchall()
            S = set(map(lambda x: x[2], records))
            exists = tablename in S
            cursor.close()
        if not exists:
            print("Creating table " + tablename)
            with self.connection.cursor() as cursor2:
                cursor2 = self.connection.cursor()
                cursor2.execute("DROP TABLE IF EXISTS " + tablename)
                self.connection.commit()
                cursor2.execute("CREATE TABLE " + tablename + " (id integer NOT NULL, idx text, t text, type text)")
                # self.connection.commit()
                cursor2.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
                # self.connection.commit()
                from LaSSI.files.ReadFileContent import ReadFileContent
                with ReadFileContent(file) as f:
                    # Notice that we don't need the csv module.
                    next(f)  # Skip the header row.
                    cursor2.copy_from(f, tablename, sep='\t')
                cursor2.execute("CREATE INDEX " + tablename + "_idx ON " + tablename + " USING GIST (t gist_trgm_ops);")
                self.connection.commit()
                cursor2.close()
        else:
            print(f"Table {tablename} already loaded!")

    def init(self, database_name, user="giacomo",
             password="omocaig",
             host="localhost",
             port="5432"):
        self.connection = psycopg2.connect(user=user,
                                           password=password,
                                           host=host,
                                           port=port,
                                           database=database_name)

    def similarity(self, table, query, score=1.0):
        query = query.replace("'", "''")
        poll = OrderedDict()
        with self.connection.cursor() as cursor:
            cursor.execute(f"""SELECT idx, similarity(t, '{query}') AS sml
                               FROM {table}
                               WHERE t % '{query}' AND similarity(t, '{query}')>={score}
                               ORDER BY sml DESC, t""")
            records = cursor.fetchall()
            for row in records:
                score = float(row[1])
                if score not in poll:
                    poll[score] = set()
                poll[score].add(row[0])
            cursor.close()
        return poll

    def typed_similarity(self, table, query, score=1.0):
        query = query.replace("'", "''")
        poll = OrderedDict()
        with self.connection.cursor() as cursor:
            cursor.execute(f"""SELECT idx, similarity(t, '{query}') AS sml, type
                               FROM {table}
                               WHERE t % '{query}' AND similarity(t, '{query}')>={score}
                               ORDER BY sml DESC, t""")
            records = cursor.fetchall()
            for row in records:
                score = float(row[1])
                if score not in poll:
                    poll[score] = set()
                poll[score].add((row[0], row[2]))
            cursor.close()
        return poll

    def morphosyntax(self, table, word, ending):
        poll = OrderedDict()
        with self.connection.cursor() as cursor:
            cursor.execute(f"""SELECT idx, t, similarity(t, '{word}') AS sml
                               FROM {table}
                               WHERE t % '{word}' AND t like '%{ending}'
                               ORDER BY sml DESC, t""")
            records = cursor.fetchall()
            for row in records:
                score = float(row[2])
                if score not in poll:
                    poll[score] = set()
                if row[1].endswith(ending):
                    poll[score].add(tuple([row[0], row[1]]))
            cursor.close()
        return poll

    @classmethod
    def instance(cls):
        if cls._instance is None:
            print('Creating new instance')
            cls._instance = cls.__new__(cls)
            # Put any initialization here.
        return cls._instance

    def __del__(self):
        self.connection.commit()
        self.connection.close()


class DBFuzzyStringMatching:
    def __init__(self, db, tablename):
        self.db = db
        self.tablename = tablename

    def fuzzyMatch(self, threshold: float, objectString: str):
        return self.db.similarity(self.tablename, objectString, score=threshold)

    def typedFuzzyMatch(self, threshold: float, objectString: str):
        return self.db.typed_similarity(self.tablename, objectString, score=threshold)


if __name__ == "__main__":
    with ReadFileContent("https://osf.io/download/a6yn8/") as r:
        next(r)
        for x in r:
            print(x)

    # c = FuzzyStringMatchDatabase.instance()
    # c.init("conceptnet")
    # print(c.morphosyntax("conceptnet", "traffic", "ed"))
    # c.create("conceptnet", "/home/giacomo/projects/similarity-pipeline/submodules/news-crawler/mini.h5_sql_input.txt")
    # c.create("geonames", "/home/giacomo/projects/similarity-pipeline/submodules/stanfordnlp_dg_server/allCountries.txt_sql_input.txt")
    # print(c.similarity("conceptnet", "giacomo", 20, .8))
    # print(c.similarity("geonames", "newcastle", 20, .8))
    # print(c.similarity("geonames", "newcastle city", 20, .8))
    # print(c.similarity("geonames", "newcastle upon tyne", 20, .8))
