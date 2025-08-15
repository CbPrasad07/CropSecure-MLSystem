
# 🌾 CropSecure-MLSystem

**An Intelligent Crop Recommendation & Weather-Integrated System**
Empowering farmers and researchers with AI-driven insights for better crop decisions.

---

## 📌 Overview

**CropSecure-MLSystem** is a **machine learning-powered agricultural decision support system** that uses **geospatial, soil, and weather data** to recommend the most suitable crops for a given location and time.
It integrates:

* **Weather Forecast API** (OpenWeather)
* **Soil Data Processing**
* **Random Forest ML Model** for accurate recommendations
* **Secure Handling of API Keys** via `.env`

---

## 🚀 Features

✅ **Dynamic Crop Recommendation** based on environmental conditions
✅ **Real-Time Weather Integration** using OpenWeather API
✅ **ML Model with High Accuracy** (Random Forest)
✅ **Easy-to-Use Interface** (Streamlit / Web App)
✅ **Scalable & Customizable** for different regions and datasets

---

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Libraries:** Pandas, NumPy, Scikit-learn, Requests, Pickle
* **API:** OpenWeather API (for real-time weather data)
* **Framework:** Streamlit / Flask (if applicable)

---

## 📂 Project Structure

```
CropSecure-MLSystem/
│-- app.py                 # Main application file
│-- requirements.txt       # Required Python libraries
│-- scaler.pkl             # Data scaler for preprocessing
│-- best_churn_model.pkl   # Trained Random Forest model
│-- .env                   # API keys (Not pushed to GitHub)
│-- README.md              # Project documentation
```

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YourUsername/CropSecure-MLSystem.git
cd CropSecure-MLSystem
```

### 2️⃣ Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Your API Key in `.env`

Create a `.env` file in the project root:

```
OPENWEATHER_API_KEY=your_api_key_here
```

### 5️⃣ Run the Application

```bash
python app.py
```

Or, if using Streamlit:

```bash
streamlit run app.py
```

---

## 🧪 Example Usage

1. Enter **location** (city/town) in the app.
2. System fetches **weather data**.
3. ML model predicts the **best crop**.
4. Recommendations displayed instantly.

---

## 📈 Future Improvements

* Integration with **satellite imagery** for pest/disease detection
* Mobile App version for offline recommendations
* Multi-language support for wider adoption

---

## 🤝 Contributing

Contributions are welcome!
Fork the repo, create a branch, commit changes, and open a pull request.

---

## 📜 License

This project is licensed under the **MIT License**.

---

Do you want me to also make you a **professional GitHub repository banner** for CropSecure-MLSystem so the repo looks premium? That will make it stand out visually.
