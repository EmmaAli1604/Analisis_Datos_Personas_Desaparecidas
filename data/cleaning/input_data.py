from ..config.config import DATA_PROCESSED
import pandas as pd

def input_data_knn(df):
    pass

def input_data_mean(df, columns):
    pass

def input_data_median(df, columns):
    pass

def input_data_mode(df, columns):
    pass

def main():
    df = pd.read_csv(DATA_PROCESSED)
    df_knn = input_data_knn(df)
    df_knn.to_csv('data/input_data/data_knn.csv', index=False)
    df_mean = input_data_mean(df, ['FECHA_NACIMIENTO', 'FECHA_DESAPARICION', 'FECHA_REGISTRO'])
    df_mean.to_csv('data/input_data/data_mean.csv', index=False)
    df_median = input_data_median(df, ['FECHA_NACIMIENTO', 'FECHA_DESAPARICION', 'FECHA_REGISTRO'])
    df_median.to_csv('data/input_data/data_median.csv', index=False)
    df_mode = input_data_mode(df, ['FECHA_NACIMIENTO', 'FECHA_DESAPARICION', 'FECHA_REGISTRO'])
    df_mode.to_csv('data/input_data/data_mode.csv', index=False)

if __name__ == "__main__":
    main()