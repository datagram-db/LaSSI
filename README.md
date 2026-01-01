# <img src="extra/lassiLogo.svg" style="height:80px; width: auto;" alt="Logo: Credits to Oliver Robert Fox (2024)" /> LaSSI

LaSSI stands for `LogicAl, Semantic, and Structure text Interpretation`. This pipeline challenges the usual learning-based approaches by making the machine directly interpret the text through logical and verifiable steps in light of recent insights on Verified Artificial Intelligence. This project originated in 2020 for a web-crawling app that was later incorporated into a broader pipeline for textual data analysis.

## Authors

* Giacomo Bergami (2020 -)
* Franco O. Saez Vander Linder (2025)
* Oliver Robert Fox (2023 - 2025)

## Installing

First, please install the dependencies required for building some of the external packages:

```bash
pip install requests==2.31.0
pip install setuptools==66.1.1
sudo apt-get install libpq-dev
```

Afterwards, you can use the usual way to install packages:

```bash
pip install .
```

Also ensure **Java** is installed.

### PostgreSQL
PostgreSQL must be installed, along with a database and user:
```bash
sudo apt install postgresql -y
sudo -u postgres psql
```
```postgresql
create database conceptnet;
create user lassi with encrypted password 'drowssap';
grant all privileges on database conceptnet to lassi;
\c conceptnet postgres
grant all on schema public to lassi;
exit
```

### Known Issues
- If PyDatagramDB fails to install, make sure all other dependencies are installed before installing.
- Currently, LaSSI has been optimised for Ubuntu, we are aware of issues with macOS which are to be fixed in the future.

## Running
The main pipeline can be run using ```main.py```. The main [datasets](https://osf.io/g5k9q/) with annotations are provided in the [orig](https://github.com/LogDS/LaSSI/tree/v2.0/test_sentences/orig) folder. The benchmarking sentences are given in [this](https://github.com/LogDS/LaSSI/tree/v2.0/test_sentences/benchmarking) other folder.
