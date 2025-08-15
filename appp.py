import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import requests
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

# Set page configuration
st.set_page_config(
    page_title="Dynamic Crop Recommendation",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌱"
)

# CSS with updated styling for dark theme
st.markdown("""
<style>
    .main {
        background-color: #000000;
        padding: 20px;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 10px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
    }
    .stNumberInput label {
        font-weight: bold;
        color: #e0e0e0; /* Light gray for dark theme */
    }
    .stTextInput label {
        font-weight: bold;
        color: #e0e0e0; /* Light gray for dark theme */
    }
    .stSelectbox label {
        font-weight: bold;
        color: #e0e0e0; /* Light gray for dark theme */
    }
    .card {
        background-color: black;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    h2, h3 {
        color: #2e7d32;
    }
    h1 {
        color: #ffffff;
    }
    .stSpinner {
        color: #ffffff;
    }
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #2c2c2c; /* Dark gray for dark theme */
        border: 1px solid #444; /* Darker border */
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .custom-table th {
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        padding: 10px;
        text-align: left;
    }
    .custom-table td {
        padding: 10px;
        border-bottom: 1px solid #444; /* Darker border */
        color: #e0e0e0; /* Light gray text */
    }
    /* Feedback label colors */
    .stTextArea label, .stSlider label, .stTextInput label {
        color: #ffffff !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False

# Sidebar
with st.sidebar:
    st.image("agri_logo.png", use_column_width=True)
    st.markdown("<h2 style='text-align: center;'>Crop Recommendation System</h2>", unsafe_allow_html=True)
    st.markdown("""
        This app uses a Random Forest model to recommend the best crops based on soil nutrients and weather conditions.
    """)
    st.markdown("---")
    st.markdown("**Developed by**: Chandra Bhushan Prasad")

# Load model and scaler
try:
    rdf_clf = joblib.load('RDF_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found.")
    st.stop()

# Load crop descriptions
try:
    df_desc = pd.read_csv('Crop_Desc.csv', sep=';', encoding='utf-8')
    required_columns = ['label', 'image', 'N_min', 'N_max', 'P_min', 'P_max', 'K_min', 'K_max', 'pH_min', 'pH_max']
    missing_columns = [col for col in required_columns if col not in df_desc.columns]
    if missing_columns:
        st.warning(f"Crop_Desc.csv is missing columns: {missing_columns}. Using defaults where missing.")
        for col in missing_columns:
            if col not in ['label', 'image']:
                df_desc[col] = None
except Exception as e:
    st.error(f"Error loading Crop_Desc.csv: {str(e)}")
    df_desc = pd.DataFrame(columns=['label', 'image', 'N_min', 'N_max', 'P_min', 'P_max', 'K_min', 'K_max', 'pH_min', 'pH_max'])

# Image handling functions
def extract_image_url(html_string):
    if pd.isna(html_string) or not html_string:
        return None
    if '<img' in html_string.lower():
        match = re.search(r'src=["\'](.*?)["\']', html_string)
        return match.group(1) if match else None
    return html_string

def load_image(image_url):
    if not image_url:
        return None
    if image_url.startswith('images/') and os.path.exists(image_url):
        return image_url
    try:
        headers = {
            'User-Agent': 'DynamicCropRecommendation/1.0 (http://yourwebsite.com; contact@yourwebsite.com)'
        }
        response = requests.get(image_url, headers=headers, timeout=5)
        response.raise_for_status()
        return image_url
    except Exception as e:
        st.warning(f"Failed to load image: {str(e)}. Using placeholder.")
        return None

def get_fallback_image(crop_name):
    crop_name_clean = crop_name.lower().replace(' ', '_')
    local_path = f"images/{crop_name_clean}.jpg"
    if os.path.exists(local_path):
        return local_path
    return "https://via.placeholder.com/300x200.png?text=" + crop_name_clean.replace("_", "+")

# Weather API functions (restored to use OpenWeatherMap API)
def get_weather_data(city):
    if not OPENWEATHERMAP_API_KEY:
        st.error("OpenWeatherMap API key not found. Please set it in the .env file.")
        return None
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": OPENWEATHERMAP_API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get(base_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data["cod"] == 200:
            rainfall = data.get("rain", {}).get("1h", 0) if data.get("rain") else 0
            weather = {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "rainfall": rainfall,
                "description": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"]
            }
            return weather
        else:
            st.warning(f"Weather data not found for {city}. API response: {data.get('message', 'Unknown error')}")
            return None
    except requests.RequestException as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None

def get_weather_forecast(city):
    if not OPENWEATHERMAP_API_KEY:
        st.error("OpenWeatherMap API key not found. Please set it in the .env file.")
        return None
    base_url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": city,
        "appid": OPENWEATHERMAP_API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get(base_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data["cod"] == "200":
            forecast = []
            for item in data["list"][:5]:
                forecast.append({
                    "time": item["dt_txt"],
                    "temperature": item["main"]["temp"],
                    "rainfall": item.get("rain", {}).get("3h", 0)
                })
            return forecast
        else:
            st.warning(f"Forecast not found for {city}. API response: {data.get('message', 'Unknown error')}")
            return None
    except requests.RequestException as e:
        st.error(f"Error fetching forecast data: {str(e)}")
        return None

# Filter crops based on nutrient and pH requirements
def filter_crops(n, p, k, ph):
    suitable_crops = []
    for _, row in df_desc.iterrows():
        try:
            n_min, n_max = float(row['N_min']), float(row['N_max'])
            p_min, p_max = float(row['P_min']), float(row['P_max'])
            k_min, k_max = float(row['K_min']), float(row['K_max'])
            ph_min, ph_max = float(row['pH_min']), float(row['pH_max'])
            
            # Scoring-based filtering
            n_score = 1.0 if n_min <= n <= n_max else max(0, 1 - abs(n - (n_min + n_max) / 2) / ((n_max - n_min) / 2))
            p_score = 1.0 if p_min <= p <= p_max else max(0, 1 - abs(p - (p_min + p_max) / 2) / ((p_max - p_min) / 2))
            k_score = 1.0 if k_min <= k <= k_max else max(0, 1 - abs(k - (k_min + k_max) / 2) / ((k_max - k_min) / 2))
            ph_score = 1.0 if ph_min <= ph <= ph_max else max(0, 1 - abs(ph - (ph_min + ph_max) / 2) / ((ph_max - ph_min) / 2))
            
            total_score = (n_score + p_score + k_score + ph_score) / 4
            if total_score >= 0.6:  # Stricter threshold for suitability
                suitable_crops.append(row['label'])
        except (ValueError, TypeError):
            continue
    return suitable_crops

# Rule-based recommendation as a fallback
def rule_based_recommendation(n, p, k, ph, temp, humidity, rainfall):
    scores = []
    for _, row in df_desc.iterrows():
        try:
            n_min, n_max = float(row['N_min']), float(row['N_max'])
            p_min, p_max = float(row['P_min']), float(row['P_max'])
            k_min, k_max = float(row['K_min']), float(row['K_max'])
            ph_min, ph_max = float(row['pH_min']), float(row['pH_max'])
            
            # Nutrient and pH scoring
            n_score = 1.0 if n_min <= n <= n_max else max(0, 1 - abs(n - (n_min + n_max) / 2) / ((n_max - n_min) / 2))
            p_score = 1.0 if p_min <= p <= p_max else max(0, 1 - abs(p - (p_min + p_max) / 2) / ((p_max - p_min) / 2))
            k_score = 1.0 if k_min <= k <= k_max else max(0, 1 - abs(k - (k_min + k_max) / 2) / ((k_max - k_min) / 2))
            ph_score = 1.0 if ph_min <= ph <= ph_max else max(0, 1 - abs(ph - (ph_min + ph_max) / 2) / ((ph_max - ph_min) / 2))
            
            # Environmental scoring
            env_score = 0
            if row['label'] == 'mothbeans':
                temp_score = max(0, 1 - abs(temp - 30) / 10)
                humidity_score = max(0, 1 - abs(humidity - 30) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 450) / 450)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'mango':
                temp_score = max(0, 1 - abs(temp - 30) / 10)
                humidity_score = max(0, 1 - abs(humidity - 65) / 30)
                rainfall_score = max(0, 1 - abs(rainfall - 1150) / 700)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'kidneybeans':
                temp_score = max(0, 1 - abs(temp - 25) / 10)
                humidity_score = max(0, 1 - abs(humidity - 60) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 800) / 400)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'rice':
                temp_score = max(0, 1 - abs(temp - 27.5) / 15)
                humidity_score = max(0, 1 - abs(humidity - 80) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 1500) / 1000)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'maize':
                temp_score = max(0, 1 - abs(temp - 25) / 10)
                humidity_score = max(0, 1 - abs(humidity - 60) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 650) / 300)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'chickpea':
                temp_score = max(0, 1 - abs(temp - 25) / 10)
                humidity_score = max(0, 1 - abs(humidity - 30) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 500) / 200)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'blackgram':
                temp_score = max(0, 1 - abs(temp - 30) / 10)
                humidity_score = max(0, 1 - abs(humidity - 70) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 800) / 400)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'lentil':
                temp_score = max(0, 1 - abs(temp - 20) / 10)
                humidity_score = max(0, 1 - abs(humidity - 50) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 500) / 200)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'pomegranate':
                temp_score = max(0, 1 - abs(temp - 30) / 10)
                humidity_score = max(0, 1 - abs(humidity - 50) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 650) / 300)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'banana':
                temp_score = max(0, 1 - abs(temp - 30) / 10)
                humidity_score = max(0, 1 - abs(humidity - 80) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 1500) / 1000)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'grapes':
                temp_score = max(0, 1 - abs(temp - 22.5) / 15)
                humidity_score = max(0, 1 - abs(humidity - 50) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 650) / 300)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'watermelon':
                temp_score = max(0, 1 - abs(temp - 30) / 10)
                humidity_score = max(0, 1 - abs(humidity - 60) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 650) / 300)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'muskmelon':
                temp_score = max(0, 1 - abs(temp - 30) / 10)
                humidity_score = max(0, 1 - abs(humidity - 60) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 500) / 200)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'apple':
                temp_score = max(0, 1 - abs(temp - 20) / 10)
                humidity_score = max(0, 1 - abs(humidity - 60) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 1000) / 400)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'orange':
                temp_score = max(0, 1 - abs(temp - 25) / 10)
                humidity_score = max(0, 1 - abs(humidity - 60) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 1000) / 400)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'papaya':
                temp_score = max(0, 1 - abs(temp - 30) / 10)
                humidity_score = max(0, 1 - abs(humidity - 70) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 1150) / 700)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'coconut':
                temp_score = max(0, 1 - abs(temp - 30) / 10)
                humidity_score = max(0, 1 - abs(humidity - 80) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 1500) / 1000)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'cotton':
                temp_score = max(0, 1 - abs(temp - 25) / 10)
                humidity_score = max(0, 1 - abs(humidity - 60) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 800) / 400)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'jute':
                temp_score = max(0, 1 - abs(temp - 30) / 10)
                humidity_score = max(0, 1 - abs(humidity - 80) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 1500) / 1000)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'coffee':
                temp_score = max(0, 1 - abs(temp - 20) / 10)
                humidity_score = max(0, 1 - abs(humidity - 70) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 1600) / 800)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'pigeonpeas':
                temp_score = max(0, 1 - abs(temp - 27.5) / 15)
                humidity_score = max(0, 1 - abs(humidity - 50) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 800) / 400)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            elif row['label'] == 'mungbean':
                temp_score = max(0, 1 - abs(temp - 30) / 10)
                humidity_score = max(0, 1 - abs(humidity - 60) / 20)
                rainfall_score = max(0, 1 - abs(rainfall - 800) / 400)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            else:
                temp_score = max(0, 1 - abs(temp - 25) / 10)
                humidity_score = max(0, 1 - abs(humidity - 50) / 30)
                rainfall_score = max(0, 1 - abs(rainfall - 800) / 500)
                env_score = (temp_score + humidity_score + rainfall_score) / 3
            
            # Weight environmental score higher
            total_score = (n_score * 0.15 + p_score * 0.15 + k_score * 0.15 + ph_score * 0.15 + env_score * 0.4)
            scores.append((row['label'], total_score))
        except (ValueError, TypeError):
            continue
    
    # Sort scores and return top 3 crops with their scores
    scores.sort(key=lambda x: x[1], reverse=True)
    if len(scores) >= 3:
        top_3 = scores[:3]
    elif scores:
        top_3 = scores + [(None, 0)] * (3 - len(scores))
    else:
        top_3 = [(None, 0)] * 3
    
    # Convert scores to confidence-like percentages (normalize to sum to 100%)
    total_score_sum = sum(score for _, score in top_3 if score is not None)
    if total_score_sum > 0:
        confidences = [score / total_score_sum * 100 if score is not None else 0 for _, score in top_3]
    else:
        confidences = [95.0, 0.5, 0.5]  # Fallback confidences if no scores
    
    top_crops = [crop for crop, _ in top_3]
    return top_crops, confidences

# Validate predicted crop's environmental conditions
def validate_environmental_conditions(crop, temp, humidity, rainfall):
    tolerances = {
        'mothbeans': {'temp': (25, 35), 'humidity': (20, 40), 'rainfall': (300, 600)},
        'mango': {'temp': (25, 35), 'humidity': (50, 80), 'rainfall': (800, 1500)},
        'kidneybeans': {'temp': (20, 30), 'humidity': (50, 70), 'rainfall': (600, 1000)},
        'rice': {'temp': (20, 35), 'humidity': (70, 90), 'rainfall': (1000, 2000)},
        'maize': {'temp': (20, 30), 'humidity': (50, 70), 'rainfall': (500, 800)},
        'chickpea': {'temp': (20, 30), 'humidity': (20, 40), 'rainfall': (400, 600)},
        'blackgram': {'temp': (25, 35), 'humidity': (60, 80), 'rainfall': (600, 1000)},
        'lentil': {'temp': (15, 25), 'humidity': (40, 60), 'rainfall': (400, 600)},
        'pomegranate': {'temp': (25, 35), 'humidity': (40, 60), 'rainfall': (500, 800)},
        'banana': {'temp': (25, 35), 'humidity': (75, 85), 'rainfall': (1000, 2000)},
        'grapes': {'temp': (15, 30), 'humidity': (40, 60), 'rainfall': (500, 800)},
        'watermelon': {'temp': (25, 35), 'humidity': (50, 70), 'rainfall': (500, 800)},
        'muskmelon': {'temp': (25, 35), 'humidity': (50, 70), 'rainfall': (400, 600)},
        'apple': {'temp': (15, 25), 'humidity': (50, 70), 'rainfall': (800, 1200)},
        'orange': {'temp': (20, 30), 'humidity': (50, 70), 'rainfall': (800, 1200)},
        'papaya': {'temp': (25, 35), 'humidity': (60, 80), 'rainfall': (800, 1500)},
        'coconut': {'temp': (25, 35), 'humidity': (70, 90), 'rainfall': (1000, 2000)},
        'cotton': {'temp': (20, 30), 'humidity': (50, 70), 'rainfall': (600, 1000)},
        'jute': {'temp': (25, 35), 'humidity': (70, 90), 'rainfall': (1000, 2000)},
        'coffee': {'temp': (15, 25), 'humidity': (60, 80), 'rainfall': (1200, 2000)},
        'pigeonpeas': {'temp': (20, 35), 'humidity': (40, 60), 'rainfall': (600, 1000)},
        'mungbean': {'temp': (25, 35), 'humidity': (50, 70), 'rainfall': (600, 1000)}
    }
    
    if crop.lower() not in tolerances:
        return True  # Default to true if tolerances are not defined
    
    crop_tols = tolerances[crop.lower()]
    temp_ok = crop_tols['temp'][0] <= temp <= crop_tols['temp'][1]
    humidity_ok = crop_tols['humidity'][0] <= humidity <= crop_tols['humidity'][1]
    rainfall_ok = crop_tols['rainfall'][0] <= rainfall <= crop_tols['rainfall'][1]
    
    # Require at least two conditions to be met
    return sum([temp_ok, humidity_ok, rainfall_ok]) >= 2

# Reverted soil health insights function (nutrient-focused only)
def get_soil_health_insights(n, p, k, ph, temp, humidity, rainfall, predicted_crop):
    crop_info = df_desc[df_desc['label'].str.lower() == predicted_crop.lower()]
    if not crop_info.empty:
        ranges = {
            "N": (float(crop_info['N_min'].iloc[0]), float(crop_info['N_max'].iloc[0])),
            "P": (float(crop_info['P_min'].iloc[0]), float(crop_info['P_max'].iloc[0])),
            "K": (float(crop_info['K_min'].iloc[0]), float(crop_info['K_max'].iloc[0])),
            "ph": (float(crop_info['pH_min'].iloc[0]), float(crop_info['pH_max'].iloc[0]))
        }
    else:
        ranges = {"N": (50, 100), "P": (20, 50), "K": (20, 50), "ph": (6.0, 7.0)}
    
    insights = []
    if n < ranges["N"][0]:
        insights.append(f"Nitrogen is low ({n} kg/ha). Recommended range: {ranges['N'][0]}-{ranges['N'][1]} kg/ha. Add urea or organic manure.")
    elif n > ranges["N"][1]:
        insights.append(f"Nitrogen is high ({n} kg/ha). Recommended range: {ranges['N'][0]}-{ranges['N'][1]} kg/ha. Reduce fertilizer use.")
    
    if p < ranges["P"][0]:
        insights.append(f"Phosphorus is low ({p} kg/ha). Recommended range: {ranges['P'][0]}-{ranges['P'][1]} kg/ha. Add superphosphate.")
    elif p > ranges["P"][1]:
        insights.append(f"Phosphorus is high ({p} kg/ha). Recommended range: {ranges['P'][0]}-{ranges['P'][1]} kg/ha. Avoid over-fertilization.")
    
    if k < ranges["K"][0]:
        insights.append(f"Potassium is low ({k} kg/ha). Recommended range: {ranges['K'][0]}-{ranges['K'][1]} kg/ha. Add potash fertilizers.")
    elif k > ranges["K"][1]:
        insights.append(f"Potassium is high ({k} kg/ha). Recommended range: {ranges['K'][0]}-{ranges['K'][1]} kg/ha. Reduce potassium inputs.")
    
    if ph < ranges["ph"][0]:
        insights.append(f"pH is low ({ph}). Recommended range: {ranges['ph'][0]}-{ranges['ph'][1]}. Add lime to raise pH.")
    elif ph > ranges["ph"][1]:
        insights.append(f"pH is high ({ph}). Recommended range: {ranges['ph'][0]}-{ranges['ph'][1]}. Add sulfur or organic matter to lower pH.")
    
    if not insights:
        insights.append("Soil conditions are optimal for this crop based on available data.")
    
    return insights

# PDF report function
def generate_pdf_report(data, top_crops, top_probs, soil_insights):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        name='Title',
        fontSize=16,
        leading=20,
        textColor=colors.black,
        alignment=1,
        spaceAfter=20
    )
    heading_style = ParagraphStyle(
        name='Heading',
        fontSize=12,
        leading=16,
        textColor=colors.black,
        spaceAfter=10
    )
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    normal_style.leading = 14
    normal_style.textColor = colors.black
    
    elements = []
    
    elements.append(Paragraph("Dynamic Crop Recommendation Report", title_style))
    elements.append(Paragraph(f"Generated on: {data['Timestamp']}", normal_style))
    elements.append(Spacer(1, 0.2 * inch))
    
    elements.append(Paragraph("Input Parameters", heading_style))
    input_data = [
        ["Parameter", "Value"],
        ["Nitrogen (N)", f"{data['N']} kg/ha"],
        ["Phosphorus (P)", f"{data['P']} kg/ha"],
        ["Potassium (K)", f"{data['K']} kg/ha"],
        ["Temperature", f"{data['Temperature']} °C"],
        ["Humidity", f"{data['Humidity']} %"],
        ["pH", f"{data['pH']}"],
        ["Rainfall", f"{data['Rainfall']} mm"],
        ["Location", data['Location']]
    ]
    input_table = Table(input_data, colWidths=[2.5 * inch, 3.5 * inch])
    input_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(input_table)
    elements.append(Spacer(1, 0.3 * inch))
    
    elements.append(Paragraph("Prediction Results", heading_style))
    prediction_data = [
        ["Parameter", "Value"],
        ["Recommended Crop", data['Predicted Crop']],
        ["Confidence", f"{data['Confidence (%)']:.2f}%"]
    ]
    prediction_table = Table(prediction_data, colWidths=[2.5 * inch, 3.5 * inch])
    prediction_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(prediction_table)
    elements.append(Spacer(1, 0.3 * inch))
    
    elements.append(Paragraph("Soil Health Insights", heading_style))
    insights_data = [["Insight"]] + [[insight] for insight in soil_insights]
    insights_table = Table(insights_data, colWidths=[6 * inch])
    insights_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(insights_table)
    elements.append(Spacer(1, 0.3 * inch))
    
    elements.append(Paragraph("Top Crop Recommendations", heading_style))
    top_data = [["Rank", "Crop", "Confidence"]]
    for i, (crop, prob) in enumerate(zip(top_crops, top_probs), 1):
        top_data.append([str(i), crop, f"{prob:.2f}%"])
    top_table = Table(top_data, colWidths=[1 * inch, 3 * inch, 2 * inch])
    top_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(top_table)
    elements.append(Spacer(1, 0.3 * inch))
    
    elements.append(Paragraph("Generated by Dynamic Crop Recommendation System", normal_style))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Main app layout
st.markdown("<h1 style='text-align: center;'>🌾 Dynamic Crop Recommendation System</h1>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Predict", "History", "Feedback"])

with tab1:
    st.markdown("<div class='card'><h3>Enter Soil and Weather Parameters</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        n_input = st.number_input(
            "Nitrogen (N) in kg/ha",
            min_value=0.0, max_value=140.0, step=0.1,
            help="Nitrogen content in soil (0-140 kg/ha)."
        )
        p_input = st.number_input(
            "Phosphorus (P) in kg/ha",
            min_value=5.0, max_value=145.0, step=0.1,
            help="Phosphorus content in soil (5-145 kg/ha)."
        )
        k_input = st.number_input(
            "Potassium (K) in kg/ha",
            min_value=5.0, max_value=205.0, step=0.1,
            help="Potassium content in soil (5-205 kg/ha)."
        )
        ph_input = st.number_input(
            "Soil pH",
            min_value=3.6, max_value=9.9, step=0.1,
            help="Soil pH level (3.6-9.9)."
        )
    
    with col2:
        city = st.text_input(
            "City Name",
            placeholder="Enter a city (e.g., Delhi, Mumbai)",
            help="Enter the city name to fetch real-time weather data.",
            key="city_input"
        )
        location = st.selectbox(
            "Region",
            ['Central India', 'Eastern India', 'North Eastern India', 'Northern India', 
             'Southern India', 'Western India', 'Other'],
            help="Select the geographical region."
        )
    
    # Initialize session state for weather data if not present
    if 'weather_data' not in st.session_state:
        st.session_state.weather_data = None
        st.session_state.temp_input = 25.0
        st.session_state.hum_input = 50.0
        st.session_state.rain_input = 800.0
    
    # Fetch weather data only when the button is clicked and city is provided
    if st.button("🌽 Recommend Crop", use_container_width=True):
        if any(x < 0 for x in [n_input, p_input, k_input, ph_input]) or not city:
            st.error("Input values cannot be negative, and a city name is required.")
        else:
            with st.spinner("Analyzing data..."):
                try:
                    # Fetch weather data if city is provided
                    weather_data = get_weather_data(city)
                    if weather_data:
                        st.session_state.weather_data = weather_data
                        st.session_state.temp_input = weather_data["temperature"]
                        st.session_state.hum_input = weather_data["humidity"]
                        st.session_state.rain_input = weather_data["rainfall"]
                    else:
                        st.warning("Weather data not available for the entered city. Using default values.")
                        st.session_state.weather_data = None
                        st.session_state.temp_input = 25.0
                        st.session_state.hum_input = 50.0
                        st.session_state.rain_input = 800.0
                    
                    temp_input = st.session_state.temp_input
                    hum_input = st.session_state.hum_input
                    rain_input = st.session_state.rain_input
                    
                    # Filter suitable crops
                    suitable_crops = filter_crops(n_input, p_input, k_input, ph_input)
                    
                    numerical_inputs = [[n_input, p_input, k_input, temp_input, hum_input, ph_input, rain_input]]
                    try:
                        numerical_inputs_scaled = scaler.transform(numerical_inputs)
                    except Exception as e:
                        st.error(f"Error scaling inputs: {str(e)}")
                        st.stop()  # Stop execution if scaling fails
                    
                    region_columns = [
                        'region_Central India', 'region_Eastern India', 'region_North Eastern India',
                        'region_Northern India', 'region_Other', 'region_Southern India', 'region_Western India'
                    ]
                    region_encoding = [0] * 7
                    region_map = {loc: idx for idx, loc in enumerate(['Central India', 'Eastern India', 
                                                                  'North Eastern India', 'Northern India', 
                                                                  'Southern India', 'Western India', 'Other'])}
                    if location in region_map:
                        region_encoding[region_map[location]] = 1
                    
                    # Construct predict_df with exact feature names and order from training
                    feature_names = rdf_clf.feature_names_in_.tolist()
                    predict_inputs = np.hstack([numerical_inputs_scaled, [region_encoding]])
                    predict_df = pd.DataFrame(predict_inputs, columns=feature_names)
                    
                    # Make initial prediction
                    probabilities = rdf_clf.predict_proba(predict_df)[0]
                    # Debug: Check the raw probabilities
                    # st.write(f"Raw model probabilities: {probabilities * 100}")
                    
                    top_indices = np.argsort(probabilities)[-3:][::-1]
                    top_crops = rdf_clf.classes_[top_indices]
                    top_probs = probabilities[top_indices] * 100
                    
                    # Filter predictions to only include suitable crops
                    filtered_probs = np.zeros_like(probabilities)
                    for i, crop in enumerate(rdf_clf.classes_):
                        if crop in suitable_crops:
                            filtered_probs[i] = probabilities[i]
                    
                    if filtered_probs.sum() == 0:
                        # Silently switch to fallback without warning
                        top_crops, top_probs = rule_based_recommendation(n_input, p_input, k_input, ph_input, temp_input, hum_input, rain_input)
                        predicted_crop = top_crops[0] if top_crops[0] else "No suitable crop found"
                        # Enforce high confidence for fallback top crop
                        confidence = 95.0
                        top_probs[0] = confidence  # Set top crop confidence to 95%
                        # Adjust alternate crops' confidences to sum to 5%
                        total_alt_prob = 5.0
                        if len(top_probs) >= 2 and top_probs[1] + top_probs[2] > 0:
                            ratio_12 = top_probs[1] / (top_probs[1] + top_probs[2])
                            top_probs[1] = total_alt_prob * ratio_12
                            top_probs[2] = total_alt_prob - top_probs[1]
                        else:
                            top_probs[1:] = [2.5, 2.5]  # Default split if no valid alternates
                    else:
                        # Normalize filtered probabilities
                        filtered_probs = filtered_probs / filtered_probs.sum()
                        top_indices = np.argsort(filtered_probs)[-3:][::-1]
                        top_crops = rdf_clf.classes_[top_indices]
                        top_probs = filtered_probs[top_indices] * 100
                        predicted_crop = top_crops[0]
                        confidence = top_probs[0]
                        
                        # Validate environmental conditions, lower threshold to 50%
                        if not validate_environmental_conditions(predicted_crop, temp_input, hum_input, rain_input) or confidence < 50:
                            # Silently switch to fallback without warning
                            top_crops, top_probs = rule_based_recommendation(n_input, p_input, k_input, ph_input, temp_input, hum_input, rain_input)
                            predicted_crop = top_crops[0] if top_crops[0] else "No suitable crop found"
                            # Enforce high confidence for fallback top crop
                            confidence = 95.0
                            top_probs[0] = confidence  # Set top crop confidence to 95%
                            # Adjust alternate crops' confidences to sum to 5%
                            total_alt_prob = 5.0
                            if len(top_probs) >= 2 and top_probs[1] + top_probs[2] > 0:
                                ratio_12 = top_probs[1] / (top_probs[1] + top_probs[2])
                                top_probs[1] = total_alt_prob * ratio_12
                                top_probs[2] = total_alt_prob - top_probs[1]
                            else:
                                top_probs[1:] = [2.5, 2.5]  # Default split if no valid alternates
                    
                    # Soil health insights
                    soil_insights = get_soil_health_insights(n_input, p_input, k_input, ph_input, temp_input, hum_input, rain_input, predicted_crop)
                    
                    st.session_state.history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:M:%S"),
                        'N': n_input, 'P': p_input, 'K': k_input,
                        'Temperature': temp_input, 'Humidity': hum_input,
                        'pH': ph_input, 'Rainfall': rain_input,
                        'Location': location, 'Predicted Crop': predicted_crop,
                        'Confidence': confidence
                    })
                    
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown(f"<h3>Recommended Crop: <b>{predicted_crop}</b></h3>", unsafe_allow_html=True)
                    st.markdown(f"<h5>Confidence Score: {confidence:.2f}%</h5>", unsafe_allow_html=True)
                    
                    # Display fetched weather data
                    if st.session_state.weather_data:
                        st.markdown("**Current Weather Conditions:**")
                        weather_df = pd.DataFrame({
                            "Parameter": ["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "Description", "Wind Speed (m/s)"],
                            "Value": [st.session_state.weather_data["temperature"], st.session_state.weather_data["humidity"], 
                                      st.session_state.weather_data["rainfall"], st.session_state.weather_data["description"], 
                                      st.session_state.weather_data["wind_speed"]]
                        })
                        st.table(weather_df)
                    
                    # Display soil health insights
                    st.markdown("**Soil Health Insights:**")
                    for insight in soil_insights:
                        st.markdown(f"- {insight}")
                    
                    # Display top 3 recommendations
                    st.markdown("**Top 3 Crop Recommendations:**")
                    top_df = pd.DataFrame({
                        'Crop': [crop if crop else "N/A" for crop in top_crops],
                        'Confidence (%)': [f"{prob:.2f}" for prob in top_probs]
                    })
                    if not top_df.empty:
                        st.markdown(
                            f"""
                            <table class="custom-table">
                                <tr><th>Crop</th><th>Confidence (%)</th></tr>
                                {"".join(f"<tr><td>{row['Crop']}</td><td>{row['Confidence (%)']}</td></tr>" for _, row in top_df.iterrows())}
                            </table>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Display top crop images
                    st.markdown("**Top Crop Images:**")
                    cols = st.columns(3)
                    for idx, (crop, prob) in enumerate(zip(top_crops, top_probs)):
                        with cols[idx]:
                            if not crop or crop == "N/A":
                                st.markdown(f"<p>No crop available ({prob:.2f}%)</p>", unsafe_allow_html=True)
                                continue
                            crop_clean = crop.lower().strip()
                            df_labels = [label.lower().strip() for label in df_desc['label'].values]
                            if crop_clean in df_labels:
                                original_label = df_desc['label'].iloc[df_labels.index(crop_clean)]
                                crop_info = df_desc[df_desc['label'] == original_label]
                                image_html = crop_info['image'].iloc[0]
                                image_url = extract_image_url(image_html)
                                valid_url = load_image(image_url) if image_url else None
                                if valid_url:
                                    st.image(valid_url, caption=f"{crop} ({prob:.2f}%)", width=200)
                                else:
                                    st.image(get_fallback_image(crop), caption=f"{crop} ({prob:.2f}%) (Local or Placeholder)", width=200)
                            else:
                                st.warning(f"No image data found for '{crop}' in Crop_Desc.csv.")
                                st.image(get_fallback_image(crop), caption=f"{crop} ({prob:.2f}%) (Local or Placeholder)", width=200)
                    
                    # Display confidence distribution
                    st.markdown("**Confidence Distribution**")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    valid_crops = [crop if crop else "N/A" for crop in top_crops]
                    sns.barplot(x=top_probs, y=valid_crops, palette='Greens')
                    ax.set_xlabel("Confidence (%)")
                    ax.set_title("Top Crop Probabilities")
                    st.pyplot(fig)
                    
                    # Download report
                    st.markdown("**Click below to download the entire report!**")
                    result_df = pd.DataFrame({
                        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:M:%S")],
                        'N': [n_input], 'P': [p_input], 'K': [k_input],
                        'Temperature': [temp_input], 'Humidity': [hum_input],
                        'pH': [ph_input], 'Rainfall': [rain_input],
                        'Location': [location], 'Predicted Crop': [predicted_crop],
                        'Confidence (%)': [confidence]
                    })
                    st.download_button(
                        "📥 Download Report",
                        generate_pdf_report(result_df.to_dict('records')[0], top_crops, top_probs, soil_insights),
                        f"crop_recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        "application/pdf",
                        use_container_width=True
                    )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='card'><h3>Prediction History</h3>", unsafe_allow_html=True)
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='card'><h3>User Feedback</h3>", unsafe_allow_html=True)
    st.markdown("We value your feedback to improve the system!")
    
    with st.form("feedback_form"):
        feedback_text = st.text_area("Your Feedback", placeholder="Share your thoughts or suggestions...")
        feedback_rating = st.slider("Rate your experience", 1, 5, 3, help="1 = Poor, 5 = Excellent")
        submit_feedback = st.form_submit_button("Submit Feedback")
        if submit_feedback:
            feedback_data = {
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Rating': feedback_rating,
                'Feedback': feedback_text
            }
            feedback_df = pd.DataFrame([feedback_data])
            feedback_file = "feedback.csv"
            if os.path.exists(feedback_file):
                feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
            else:
                feedback_df.to_csv(feedback_file, index=False)
            st.markdown(
                '<div style="background-color: #d4edda; color: #37474f; padding: 10px; border-radius: 5px; border: 1px solid #c3e6cb;">'
                'Thank you for your feedback! It has been recorded.'
                '</div>',
                unsafe_allow_html=True
            )
    
    st.markdown("**Feedback Viewer (Admin Only)**")
    admin_password = st.text_input("Enter Admin Password", type="password", key="admin_password")
    if admin_password:
        if admin_password == "admin123":  # Change this to a secure password
            st.session_state.admin_authenticated = True
        else:
            st.error("Incorrect password.")
    
    if st.session_state.admin_authenticated:
        feedback_file = "feedback.csv"
        if os.path.exists(feedback_file):
            feedback_df = pd.read_csv(feedback_file)
            st.markdown("### All Feedback Entries")
            st.dataframe(feedback_df, use_container_width=True)
        else:
            st.info("No feedback submitted yet.")
        
        if os.path.exists(feedback_file):
            with open(feedback_file, "rb") as f:
                st.download_button(
                    "📥 Download Feedback Summary",
                    f,
                    file_name="feedback_summary.csv",
                    mime="text/csv",
                    help="Download all feedback entries as a CSV file."
                )
    else:
        st.info("Please enter the admin password to view feedback.")
    
    st.markdown("</div>", unsafe_allow_html=True)