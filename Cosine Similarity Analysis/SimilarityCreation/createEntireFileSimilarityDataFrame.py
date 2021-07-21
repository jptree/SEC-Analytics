import pandas as pd
import os

if __name__ == "__main__":
    _, _, directories = next(os.walk('../EntireFile/Cosine/TfIdf/Quarterly'))

    df_all = pd.DataFrame()
    for d in directories:
        df = pd.read_csv(f'EntireFile/Quarterly/{d}', header=None, names=['date', 'cik', 'similarity'])
        df_all = df_all.append(df)

    print(df_all)
    df_all.to_csv('EntireFileSimilarity.csv')
