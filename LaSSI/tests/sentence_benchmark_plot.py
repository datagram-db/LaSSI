import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('/home/campus.ncl.ac.uk/b9063849/PycharmProjects/LaSSI/benchmark_results_200.csv')

    df.groupby(by='Sentence length').mean().to_csv('/home/campus.ncl.ac.uk/b9063849/PycharmProjects/LaSSI/benchmark_results_200_grouped.csv')