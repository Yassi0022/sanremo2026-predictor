from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Carica il modello e il file CSV con le predizioni che hai appena salvato
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
df_pred = pd.read_csv("data/predictions.csv")

# Home Page
@app.route("/")
def home():
    # Trasforma la top 15 in una lista di dizionari da passare al template HTML
    top15 = df_pred.head(15)[["Rank", "Artista", "Canzone", "Prob_Vittoria"]].to_dict(orient="records")
    return render_template("index.html", top15=top15)

# API per il Grafico JS
@app.route("/api/top")
def api_top():
    top10 = df_pred.head(10)[["Artista", "Prob_Vittoria"]].to_dict(orient="records")
    return jsonify(top10)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
