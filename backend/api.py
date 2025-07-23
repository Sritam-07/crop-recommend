from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

final_pipe = joblib.load('crop_recommender.pkl')
le_target = joblib.load('label_encoder.pkl')

def feature_engineer(df):
    df['NPK'] = (df['N'] + df['P'] + df['K']) / 3
    df['THI'] = df['temperature'] * df['humidity'] / 100
    df['rainfall_level'] = pd.cut(df['rainfall'],
                                 bins=[0, 50, 100, 200, 300],
                                 labels=['Low', 'Medium', 'High', 'Very High'])
    def ph_category(p):
        if p < 5.5:
            return 'Acidic'
        elif p <= 7.5:
            return 'Neutral'
        else:
            return 'Alkaline'
    df['ph_category'] = df['ph'].apply(ph_category)
    df['temp_rain_interaction'] = df['temperature'] * df['rainfall']
    df['ph_rain_interaction'] = df['ph'] * df['rainfall']
    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        farm_df = pd.DataFrame([data])
        farm_df_fe = feature_engineer(farm_df)

        prediction_encoded = final_pipe.predict(farm_df_fe)
        predicted_crop = le_target.inverse_transform(prediction_encoded)[0]

        probabilities = final_pipe.predict_proba(farm_df_fe)[0]
        crop_probabilities = dict(zip(le_target.classes_, probabilities.tolist()))

        return jsonify({
            'recommended_crop': predicted_crop,
            'crop_probabilities': crop_probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)
