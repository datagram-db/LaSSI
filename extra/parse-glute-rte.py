import pandas
import yaml

if __name__ == "__main__":
    file = "/home/giacomo/projects/LaSSI/test_sentences/glue-rte.csv"
    pd = pandas.read_csv(file)
    pd = pd.reindex(pd.sentence1.str.len().sort_values().index)
    with open("../test_sentences/real_data/glue-rte_answers.yaml", "w") as file:
        yaml.safe_dump(list(pd["sentence2"]), file)
    with open("../test_sentences/real_data/glue-rte_questions.yaml", "w") as file:
        yaml.safe_dump(list(pd["sentence1"]), file)