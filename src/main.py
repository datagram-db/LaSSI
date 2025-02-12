from ontology_generator import generate
from config import config
import argparse

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument("-c", "--conceptnet", type=str, required=True, help="Path to coneptnet csv file")
    # parser.add_argument("-j", "--wiktionary", type=str, required=True, help="Path to wiktionary json file")
    #
    # args = parser.parse_args()

    generate(config["conceptnet_csv"], config["wiktionary_json"], 3000)