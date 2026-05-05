from components.simulador_umbral import simulador_umbral
from src.models.forecasting import main_forecasting

def page_simulador_umbral():
    y_test, y_predictions, model, X_test, data_raw = main_forecasting()
    simulador_umbral(model, X_test, y_test)

page_simulador_umbral()