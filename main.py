from src.ontology_generator import generate
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Adding required arguments for paths
    parser.add_argument("-c", "--conceptnet", type=str, required=True, help="Path to coneptnet csv file")
    parser.add_argument("-j", "--wiktionary", type=str, required=True, help="Path to wiktionary json file")

    args = parser.parse_args()
    generate(args.conceptnet, args.wiktionary)