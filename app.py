from flask import Flask, render_template, request, jsonify
import csv
import joblib


app = Flask(__name__)

# Carica il modello e il file CSV con le predizioni che hai appena salvato
def load_predictions():
    data = []
    with open('data/predictions.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['Prob_Vittoria'] = float(row['Prob_Vittoria'])
            data.append(row)
    return data

df_pred = load_predictions()

# Home Page
@app.route("/")
def home():
    # Trasforma la top 15 in una lista di dizionari da passare al template HTML
    top15 = df_pred[:15]
    return render_template("index.html", top15=top15)

# API per il Grafico JS
@app.route("/api/top")
def api_top():
    top10 = [{"Artista": row["Artista"], "Prob_Vittoria": row["Prob_Vittoria"]} for row in df_pred[:10]]
    return jsonify(top10)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
