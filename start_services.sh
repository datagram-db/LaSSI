#!/usr/bin/bash
pushd submodules/news-crawler
pip3 install -r requirements.txt
popd
pushd submodules/stanford_dg_server
mvn compile exec:java -Dexec.mainClass="uk.ncl.giacomobergami.Main" & # Starting the java service
popd
