#!/usr/bin/bash
pushd submodules/news-crawler
chmod +x main.py
pip3 install -r requirements.txt
popd
mvn -f submodules/stanfordnlp_dg_server/pom.xml compile exec:java -Dexec.mainClass="uk.ncl.giacomobergami.Main" & # Starting the java service
python3 submodules/news-crawler/main.py                               #
