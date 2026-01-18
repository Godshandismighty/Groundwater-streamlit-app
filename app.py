import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Groundwater Level Prediction",
    page_icon="💧",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #145a8c;
    }
    h1 {
        color: #1f77b4;
    }
    .prediction-box {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stat-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load and train model
@st.cache_resource
def load_and_train_model():
    # Load the CSV file
    df = pd.read_csv('groundwater_level_data_1000_rows.csv')
    
    # Prepare features and target
    X = df[['rainfall_mm', 'temperature_c', 'humidity_percent', 'soil_moisture']]
    y = df['groundwater_level_m']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Calculate statistics
    stats = {
        'rainfall_range': (df['rainfall_mm'].min(), df['rainfall_mm'].max()),
        'temp_range': (df['temperature_c'].min(), df['temperature_c'].max()),
        'humidity_range': (df['humidity_percent'].min(), df['humidity_percent'].max()),
        'soil_range': (df['soil_moisture'].min(), df['soil_moisture'].max()),
        'groundwater_range': (df['groundwater_level_m'].min(), df['groundwater_level_m'].max()),
        'avg_groundwater': df['groundwater_level_m'].mean()
    }
    
    return model, scaler, df, stats

# Load model and data
model, scaler, data, stats = load_and_train_model()

# Header
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("💧")
with col2:
    st.title("Groundwater Level Prediction")

st.markdown("---")
st.write("Enter environmental parameters to predict groundwater level.")

# Input section
st.subheader("📊 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=float(stats['rainfall_range'][1]),
        value=float(stats['rainfall_range'][1] / 2),
        step=0.5,
        help=f"Range: {stats['rainfall_range'][0]:.2f} - {stats['rainfall_range'][1]:.2f} mm"
    )
    
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=float(stats['temp_range'][0]),
        max_value=float(stats['temp_range'][1]),
        value=float(stats['temp_range'][1] / 2),
        step=0.5,
        help=f"Range: {stats['temp_range'][0]:.2f} - {stats['temp_range'][1]:.2f} °C"
    )

with col2:
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(stats['humidity_range'][1] / 2),
        step=1.0,
        help=f"Range: {stats['humidity_range'][0]:.2f} - {stats['humidity_range'][1]:.2f} %"
    )
    
    soil_moisture = st.number_input(
        "Soil Moisture",
        min_value=0.0,
        max_value=float(stats['soil_range'][1]),
        value=float(stats['soil_range'][1] / 2),
        step=0.1,
        help=f"Range: {stats['soil_range'][0]:.2f} - {stats['soil_range'][1]:.2f}"
    )

st.markdown("---")

# Predict button
if st.button("🔮 Predict Groundwater Level"):
    # Prepare input data
    input_data = np.array([[rainfall, temperature, humidity, soil_moisture]])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Display results
    st.markdown("### 📈 Prediction Results")
    
    # Main prediction box
    st.markdown(f"""
        <div class="prediction-box">
            <h2 style="margin: 0; color: #1f77b4;">Predicted Groundwater Level</h2>
            <h1 style="margin: 10px 0; color: #1f77b4; font-size: 48px;">{prediction:.2f} meters</h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Statistics comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="stat-box">
                <h4>Average Level</h4>
                <h3 style="color: #1f77b4;">{stats['avg_groundwater']:.2f} m</h3>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stat-box">
                <h4>Minimum Level</h4>
                <h3 style="color: #ff7f0e;">{stats['groundwater_range'][0]:.2f} m</h3>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="stat-box">
                <h4>Maximum Level</h4>
                <h3 style="color: #2ca02c;">{stats['groundwater_range'][1]:.2f} m</h3>
            </div>
        """, unsafe_allow_html=True)
    
    # Visualization
    st.markdown("### 📊 Visualization")
    
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Groundwater Level (m)"},
        delta={'reference': stats['avg_groundwater']},
        gauge={
            'axis': {'range': [None, stats['groundwater_range'][1]]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, stats['groundwater_range'][1] * 0.33], 'color': "lightgray"},
                {'range': [stats['groundwater_range'][1] * 0.33, stats['groundwater_range'][1] * 0.66], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': stats['avg_groundwater']
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Input vs Prediction comparison
    st.markdown("### 📋 Input Summary")
    
    input_df = pd.DataFrame({
        'Parameter': ['Rainfall', 'Temperature', 'Humidity', 'Soil Moisture'],
        'Value': [f"{rainfall:.2f} mm", f"{temperature:.2f} °C", f"{humidity:.2f} %", f"{soil_moisture:.2f}"],
        'Status': ['✓', '✓', '✓', '✓']
    })
    
    st.dataframe(input_df, use_container_width=True, hide_index=True)

# Sidebar with additional info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app predicts groundwater levels based on environmental parameters using machine learning.
    
    **Model Information:**
    - Algorithm: Random Forest Regressor
    - Training Data: 1000 observations
    - Features: Rainfall, Temperature, Humidity, Soil Moisture
    """)
    
    st.markdown("---")
    
    st.header("📊 Dataset Statistics")
    st.write(f"**Total Records:** 1000")
    st.write(f"**Avg Groundwater Level:** {stats['avg_groundwater']:.2f} m")
    st.write(f"**Range:** {stats['groundwater_range'][0]:.2f} - {stats['groundwater_range'][1]:.2f} m")
    
    st.markdown("---")
    
    if st.checkbox("Show Data Sample"):
        st.dataframe(data.head(10))
    
    if st.checkbox("Show Feature Importance"):
        feature_importance = pd.DataFrame({
            'Feature': ['Rainfall', 'Temperature', 'Humidity', 'Soil Moisture'],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                     title='Feature Importance')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>Groundwater Level Prediction System | Powered by Machine Learning</p>
    </div>
""", unsafe_allow_html=True)
