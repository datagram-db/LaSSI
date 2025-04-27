import pandas

if __name__ == "__main__":
    file = "/home/giacomo/Scrivania/LaSSI/cm.csv_12.csv"
    df = pandas.read_csv(file, index_col=0)
    df.transpose().to_latex("/home/giacomo/Scrivania/LaSSI/cm.tex")
    print(df)