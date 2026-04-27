# streamlit/config/lecture.py
import sys
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.config.config import DATA_RAW, DATA_INPUT, DATA_PROCESSED

def load_lecture_data_raw():
    return pd.read_csv(DATA_RAW)

def load_lecture_data_inputed():
    return pd.read_csv(DATA_INPUT)

def load_lecture_data_processed():
    return pd.read_csv(DATA_PROCESSED)