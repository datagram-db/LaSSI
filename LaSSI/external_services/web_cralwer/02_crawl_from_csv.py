#!/usr/bin/env python3
__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2020, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"
## This script allows to download the files even if the main process just wrote the csv without downloading the data from the servers
import sys, argparse, csv, os, json
from os import path
from newsplease import NewsPlease

database = dict()

def craw_from_file(filename):
    print(filename)
    print('-------------------')
    print('-------------------')
    dkey = filename[:-15]
    dkey_dbfile = dkey+".json"
    dkey_db = {}
    if path.exists(dkey_dbfile):
        dkey_db = json.load(open(dkey_dbfile))
    dkey_db_nextKey = 1 
    if len(dkey_db) > 0:
        dkey_db_nextKey = int(max(dkey_db,key=int))+1
    with open(filename, 'r') as csvfile:
    	reader = csv.reader(csvfile)
    	for row in reader:
    	        if not (dkey in database):
    	            database[dkey] = set()
    	        else:
    	            database[dkey] = set(database[dkey])
    	        if not (row[0] in database[dkey]):
    	            try:
    	               database[dkey].add(row[0])
    	               dkey_db[str(dkey_db_nextKey)] = NewsPlease.from_url(row[0]).get_serializable_dict()
    	               print("Article #"+str(dkey_db_nextKey)+" was parsed")
    	               dkey_db_nextKey = dkey_db_nextKey+1
    	            except Exception as e:
    	               print("Error: ''"+str(e)+"'' for article at ''"+row[0]+"''! downloading it as a simple text file...")
    json.dump(dkey_db,open(dkey_dbfile,'w'))
    dump_database('db.json')


def load_database(filename):
    global database
    if path.exists(filename):
        with open(filename) as json_file:
            data = json.load(json_file)
            for k in data:
                data[k] = set(data[k])
            database = data

def dump_database(filename):
    for k in database:
        database[k] = list(database[k])
    json.dump(database,open(filename,'w'))

def get_all_csvs():
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for f in files:
        if (f.endswith('.csv')):
            craw_from_file(f)
         

if __name__ == '__main__':
    load_database('db.json')
    get_all_csvs()
    dump_database('db.json')
