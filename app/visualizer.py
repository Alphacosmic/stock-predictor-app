import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_arima_forecast(actual_series, forecast_series, title="ARIMA Forecast vs Actual"):
    plt.figure(figsize=(10, 5))
    plt.plot(actual_series.index, actual_series.values, label="Actual", marker='o')
    plt.plot(forecast_series.index, forecast_series.values, label="Forecast", linestyle='--', marker='x')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_xgboost_confusion_matrix(y_true, y_pred, title="XGBoost Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, title="XGBoost Feature Importance"):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    sorted_features = [feature_names[i] for i in indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance[indices], y=sorted_features, palette="viridis")
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()