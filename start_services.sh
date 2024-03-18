#!/usr/bin/bash
pushd submodules/news-crawler
chmod +x main.py
pip3 install -r requirements.txt
popd
pushd submodules/stanford_dg_server
mvn compile exec:java -Dexec.mainClass="uk.ncl.giacomobergami.Main" -Dexec.workingdir="../../" & # Starting the java service
popd
python3 submodules/news-crawler/main.py                               #
