import pandas

if __name__ == "__main__":
    file = "/home/giacomo/Scrivania/LaSSI/n.csv"
    df = pandas.read_csv(file, index_col=0)
    df = df.transpose()
    df.to_latex("/home/giacomo/Scrivania/LaSSI/n.tex")
    print(df)