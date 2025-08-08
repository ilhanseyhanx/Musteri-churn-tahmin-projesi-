async function makePrediction() {
    let data = {
        gender: document.getElementById("gender").value,
        SeniorCitizen: parseInt(document.getElementById("SeniorCitizen").value),
        Partner: document.getElementById("Partner").value,
        Dependents: document.getElementById("Dependents").value,
        tenure: parseInt(document.getElementById("tenure").value),
        PhoneService: document.getElementById("PhoneService").value,
        MultipleLines: document.getElementById("MultipleLines").value,
        InternetService: document.getElementById("InternetService").value,
        OnlineSecurity: document.getElementById("OnlineSecurity").value,
        OnlineBackup: document.getElementById("OnlineBackup").value,
        DeviceProtection: document.getElementById("DeviceProtection").value,
        TechSupport: document.getElementById("TechSupport").value,
        StreamingTV: document.getElementById("StreamingTV").value,
        StreamingMovies: document.getElementById("StreamingMovies").value,
        Contract: document.getElementById("Contract").value,
        PaperlessBilling: document.getElementById("PaperlessBilling").value,
        MonthlyCharges: parseFloat(document.getElementById("MonthlyCharges").value),
        TotalCharges: parseFloat(document.getElementById("TotalCharges").value),
        PaymentMethod: document.getElementById("PaymentMethod").value
    };

    try {
        let response = await fetch('https://musteri-churn-tahmin-projesi-production.up.railway.app/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        let result = await response.json();

        document.getElementById("result").innerHTML =
            "<h3>Tahmin Sonucu:</h3>" +
            "<p>Churn: " + (result.Churn_Prediction == 1 ? "AYRILACAK" : "KALACAK") + "</p>" +
            "<p>Risk: %" + (result.Churn_Probability * 100).toFixed(1) + "</p>";
    } catch (error) {
        document.getElementById("result").innerHTML = "<p>Hata: " + error + "</p>";
    }
}

document.querySelector('button').addEventListener('click', makePrediction);