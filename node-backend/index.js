const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();

app.use(cors());
app.use(express.json());

app.post('/api/predict', async (req, res) => {
  try {
    const response = await axios.post('http://localhost:5000/predict', req.body);
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: 'Prediction service error' });
  }
});

const PORT = 3001;
app.listen(PORT, () => {
  console.log(`Express server running on port ${PORT}`);
});
