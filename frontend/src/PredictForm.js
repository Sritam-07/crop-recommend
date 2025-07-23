import React, { useState } from 'react';
import axios from 'axios';

const PredictForm = () => {
  const [formData, setFormData] = useState({
    N: '',
    P: '',
    K: '',
    temperature: '',
    humidity: '',
    ph: '',
    rainfall: ''
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFormData({...formData, [e.target.name]: e.target.value});
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const payload = {};
      Object.keys(formData).forEach(key => {
        payload[key] = parseFloat(formData[key]);
      });

      const response = await axios.post('http://localhost:3001/api/predict', payload);
      setResult(response.data);
    } catch (err) {
      setError('Failed to get prediction');
    }
    setLoading(false);
  }

  return (
    <div style={{ maxWidth: 500, margin: 'auto' }}>
      <h2>Crop Recommendation</h2>
      <form onSubmit={handleSubmit}>
        {Object.keys(formData).map(key => (
          <div key={key} style={{ marginBottom: 12 }}>
            <label htmlFor={key} style={{ display: 'block', fontWeight: 'bold' }}>
              {key.toUpperCase()}:
            </label>
            <input
              type="number"
              step="any"
              id={key}
              name={key}
              value={formData[key]}
              onChange={handleChange}
              required
              style={{ width: '100%', padding: '8px' }}
            />
          </div>
        ))}
        <button type="submit" disabled={loading}>
          {loading ? 'Predicting...' : 'Get Recommendation'}
        </button>
      </form>

      {error && <p style={{ color: 'red' }}>{error}</p>}

      {result && (
        <div style={{ marginTop: 20 }}>
          <h3>Recommended Crop: {result.recommended_crop}</h3>
          <h4>Probabilities:</h4>
          <ul>
            {Object.entries(result.crop_probabilities).map(([crop, prob]) => (
              <li key={crop}>{crop}: {(prob * 100).toFixed(2)}%</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default PredictForm;
