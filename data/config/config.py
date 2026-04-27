# data/config/config.py
import os

# Raíz del proyecto: sube 2 niveles desde data/config/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_RAW       = os.path.join(ROOT, "data", "raw",       "data_secretariado.csv")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed", "data_processed.csv")
DATA_INPUT     = os.path.join(ROOT, "data", "processed", "data_imputed.csv")