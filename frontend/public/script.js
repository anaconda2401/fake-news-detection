document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const textArea = document.getElementById('news-text');
    const resultsSection = document.getElementById('results');
    const spinner = document.getElementById('spinner');
    const errorBox = document.getElementById('error-box');

    // Ensemble DOM Elements
    const cardEnsemble = document.getElementById('card-ensemble');
    const labelEnsemble = document.getElementById('res-ensemble-label');
    const barEnsemble = document.getElementById('bar-ensemble');
    const confEnsemble = document.getElementById('conf-ensemble');
    
    // DistilBERT DOM Elements
    const cardDistilbert = document.getElementById('card-distilbert');
    const labelDistilbert = document.getElementById('res-distilbert-label');
    const barDistilbert = document.getElementById('bar-distilbert');
    const confDistilbert = document.getElementById('conf-distilbert');

    analyzeBtn.addEventListener('click', async () => {
        const text = textArea.value.trim();
        
        if (!text) {
            alert("Please enter some text to analyze.");
            return;
        }

        // Reset UI State
        resultsSection.classList.add('hidden');
        errorBox.classList.add('hidden');
        spinner.classList.remove('hidden');
        analyzeBtn.disabled = true;

        // Reset old classes
        cardEnsemble.className = "result-card";
        cardDistilbert.className = "result-card";
        labelEnsemble.className = "prediction-value";
        labelDistilbert.className = "prediction-value";
        barEnsemble.style.width = '0%';
        barDistilbert.style.width = '0%';

        try {
            // Call Node.js Backend API
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || `HTTP Error ${response.status}`);
            }

            const data = await response.json();
            
            // Populate Results
            populateCard(
                data.ensemble, 
                cardEnsemble, 
                labelEnsemble, 
                barEnsemble, 
                confEnsemble
            );

            populateCard(
                data.distilbert, 
                cardDistilbert, 
                labelDistilbert, 
                barDistilbert, 
                confDistilbert
            );

            // Show results section smoothly
            resultsSection.classList.remove('hidden');

            // Trigger reflow to restart CSS animation
            void resultsSection.offsetWidth;

            // Animate confidence bars after a short delay
            setTimeout(() => {
                barEnsemble.style.width = `${(data.ensemble.confidence * 100).toFixed(0)}%`;
                barDistilbert.style.width = `${(data.distilbert.confidence * 100).toFixed(0)}%`;
            }, 100);

        } catch (error) {
            console.error(error);
            errorBox.textContent = `Analysis failed: ${error.message}`;
            errorBox.classList.remove('hidden');
        } finally {
            spinner.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    });

    function populateCard(modelData, cardEl, labelEl, barEl, confEl) {
        // modelData contains { prediction: 0/1, label: "Real News"/"Fake News", confidence: 0.0 - 1.0 }
        
        labelEl.textContent = modelData.label;
        const confPercent = (modelData.confidence * 100).toFixed(1);
        confEl.textContent = `${confPercent}%`;

        if (modelData.prediction === 1) { // 1 = Fake
            labelEl.classList.add('is-fake');
            cardEl.classList.add('fake-card');
        } else {
            labelEl.classList.add('is-real');
            cardEl.classList.add('real-card');
        }
    }
});
