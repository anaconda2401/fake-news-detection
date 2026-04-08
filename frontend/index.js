const express = require('express');
const cors = require('cors');
const axios = require('axios');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_API_URL = 'http://127.0.0.1:5000/predict';

app.use(cors());
app.use(express.json());

app.use(express.static(path.join(__dirname, 'public')));

app.post('/api/predict', async (req, res) => {
    try {
        const { text } = req.body;
        if (!text) {
            return res.status(400).json({ error: 'Text is required' });
        }

        const pythonResponse = await axios.post(PYTHON_API_URL, { text });
        res.json(pythonResponse.data);

    } catch (error) {
        console.error("Error connecting to Python service:", error.message);
        res.status(500).json({ 
            error: 'Backend ML service is currently unavailable.',
            details: error.message
        });
    }
});

app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`Node.js Web Server is running on http://localhost:${PORT}`);
    console.log(`Expecting Python ML Service on ${PYTHON_API_URL}`);
});
