from ..config.config import DATA_RAW
import pandas as pd

def load_data():
    df = pd.read_csv(DATA_RAW)
    return df

def print_data_info(df):
    print("="*80)
    print("Data loaded successfully.")
    print("="*80)
    print(df.head())
    print("="*80)
    print("Data Info:")
    print(df.info())
    print("="*80)
    print("Data Description:")
    print(df.describe())
    print("="*80)
    print("Missing Values:")
    print(df.isnull().sum())
    print("="*80)
    
def main():
    df = load_data()
    
    
if __name__ == "__main__":
    main()