"""
🌫️ PM2.5 Air Quality Forecasting Application
============================================
A Streamlit web application for predicting PM2.5 air quality levels
using a Hybrid Transformer-LSTM Deep Learning Model.

Author: Sheryar Sher
Project: Attention-Based Deep Learning for Hyper-Local Air Quality Prediction
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, MultiHeadAttention, Dense, Dropout, LayerNormalization
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="PM2.5 Air Quality Forecaster",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1E88E5;
        --secondary-color: #43A047;
        --warning-color: #FB8C00;
        --danger-color: #E53935;
        --background-dark: #0E1117;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        color: #00d4ff;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 10px rgba(0,212,255,0.5);
    }
    
    .main-header p {
        color: #a0a0a0;
        font-size: 1.1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2f 0%, #2d2d44 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border: 1px solid #333;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #888;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* AQI Level badges */
    .aqi-good { color: #4CAF50; }
    .aqi-moderate { color: #FFEB3B; }
    .aqi-unhealthy-sensitive { color: #FF9800; }
    .aqi-unhealthy { color: #F44336; }
    .aqi-very-unhealthy { color: #9C27B0; }
    .aqi-hazardous { color: #7B1FA2; }
    
    /* Info boxes */
    .info-box {
        background: rgba(30, 136, 229, 0.1);
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Prediction result box */
    .prediction-box {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        border: 2px solid;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #1a1a2e;
    }
    
    /* Feature input sections */
    .feature-section {
        background: rgba(255,255,255,0.02);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #333;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    ::-webkit-scrollbar-thumb {
        background: #444;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CUSTOM KERAS LAYERS (Required for model loading)
# ============================================================================

class TransformerEncoderBlock(Layer):
    """Transformer Encoder Block with Multi-Head Self-Attention."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate
        )
        self.ffn = keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training=False, return_attention=False):
        if return_attention:
            attn_output, attn_weights = self.attention(
                inputs, inputs, training=training, return_attention_scores=True
            )
        else:
            attn_output = self.attention(inputs, inputs, training=training)
            attn_weights = None
        
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output = self.layernorm2(out1 + ffn_output)
        
        if return_attention:
            return output, attn_weights
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate
        })
        return config


class PositionalEncoding(Layer):
    """Sinusoidal Positional Encoding Layer."""
    
    def __init__(self, max_seq_len, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.pos_encoding = self._create_positional_encoding()
    
    def _create_positional_encoding(self):
        positions = np.arange(self.max_seq_len)[:, np.newaxis]
        dims = np.arange(self.embed_dim)[np.newaxis, :]
        angles = positions / np.power(10000, (2 * (dims // 2)) / self.embed_dim)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        pos_encoding = angles[np.newaxis, :, :]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_seq_len': self.max_seq_len,
            'embed_dim': self.embed_dim
        })
        return config


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

FEATURE_COLUMNS = ['PM2.5', 'PM10', 'TEMP', 'PRES', 'DEWP', 'WSPM', 'NO2', 'SO2', 'CO', 'O3']
INPUT_WINDOW = 48
OUTPUT_HORIZON = 12

# Feature descriptions and units
FEATURE_INFO = {
    'PM2.5': {'name': 'PM2.5', 'unit': 'μg/m³', 'description': 'Fine Particulate Matter (≤2.5μm)', 'min': 0, 'max': 500, 'default': 50},
    'PM10': {'name': 'PM10', 'unit': 'μg/m³', 'description': 'Coarse Particulate Matter (≤10μm)', 'min': 0, 'max': 600, 'default': 80},
    'TEMP': {'name': 'Temperature', 'unit': '°C', 'description': 'Ambient Temperature', 'min': -20, 'max': 45, 'default': 20},
    'PRES': {'name': 'Pressure', 'unit': 'hPa', 'description': 'Atmospheric Pressure', 'min': 980, 'max': 1050, 'default': 1013},
    'DEWP': {'name': 'Dew Point', 'unit': '°C', 'description': 'Dew Point Temperature', 'min': -40, 'max': 30, 'default': 10},
    'WSPM': {'name': 'Wind Speed', 'unit': 'm/s', 'description': 'Wind Speed', 'min': 0, 'max': 20, 'default': 2},
    'NO2': {'name': 'NO₂', 'unit': 'μg/m³', 'description': 'Nitrogen Dioxide', 'min': 0, 'max': 300, 'default': 40},
    'SO2': {'name': 'SO₂', 'unit': 'μg/m³', 'description': 'Sulfur Dioxide', 'min': 0, 'max': 200, 'default': 15},
    'CO': {'name': 'CO', 'unit': 'μg/m³', 'description': 'Carbon Monoxide', 'min': 0, 'max': 10000, 'default': 800},
    'O3': {'name': 'O₃', 'unit': 'μg/m³', 'description': 'Ozone', 'min': 0, 'max': 300, 'default': 50}
}

# AQI Categories based on PM2.5
AQI_CATEGORIES = [
    {'range': (0, 12), 'level': 'Good', 'color': '#4CAF50', 'emoji': '😊', 'advice': 'Air quality is satisfactory. Enjoy outdoor activities!'},
    {'range': (12.1, 35.4), 'level': 'Moderate', 'color': '#FFEB3B', 'emoji': '😐', 'advice': 'Acceptable air quality. Unusually sensitive people should consider limiting prolonged outdoor exertion.'},
    {'range': (35.5, 55.4), 'level': 'Unhealthy for Sensitive Groups', 'color': '#FF9800', 'emoji': '😷', 'advice': 'Members of sensitive groups may experience health effects. General public is less likely to be affected.'},
    {'range': (55.5, 150.4), 'level': 'Unhealthy', 'color': '#F44336', 'emoji': '🤢', 'advice': 'Everyone may begin to experience health effects. Sensitive groups may experience more serious effects.'},
    {'range': (150.5, 250.4), 'level': 'Very Unhealthy', 'color': '#9C27B0', 'emoji': '🚨', 'advice': 'Health alert: everyone may experience more serious health effects. Avoid outdoor activities.'},
    {'range': (250.5, 500), 'level': 'Hazardous', 'color': '#7B1FA2', 'emoji': '☠️', 'advice': 'Health warning of emergency conditions. The entire population is likely to be affected. Stay indoors!'}
]


def get_aqi_category(pm25_value):
    """Get AQI category based on PM2.5 value."""
    for cat in AQI_CATEGORIES:
        if cat['range'][0] <= pm25_value <= cat['range'][1]:
            return cat
    return AQI_CATEGORIES[-1]  # Hazardous for values > 500


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model_and_scalers():
    """Load the trained model and scalers."""
    try:
        # Get the directory of the current script
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, 'model')
        
        # Register custom layers
        custom_objects = {
            'TransformerEncoderBlock': TransformerEncoderBlock,
            'PositionalEncoding': PositionalEncoding
        }
        
        # Load model
        model = keras.models.load_model(
            os.path.join(model_path, 'hybrid_t_lstm_final.keras'),
            custom_objects=custom_objects
        )
        
        # Load scalers
        with open(os.path.join(model_path, 'feature_scaler.pkl'), 'rb') as f:
            feature_scaler = pickle.load(f)
        
        with open(os.path.join(model_path, 'target_scaler.pkl'), 'rb') as f:
            target_scaler = pickle.load(f)
        
        return model, feature_scaler, target_scaler, True
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, False


def predict_pm25(model, feature_scaler, target_scaler, input_data):
    """Make PM2.5 predictions using the model."""
    try:
        # Scale input data
        input_scaled = feature_scaler.transform(input_data)
        
        # Reshape for model input (batch, time_steps, features)
        input_reshaped = input_scaled.reshape(1, INPUT_WINDOW, len(FEATURE_COLUMNS))
        
        # Make prediction
        prediction_scaled = model.predict(input_reshaped, verbose=0)
        
        # Inverse transform predictions
        prediction = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
        
        return prediction.flatten()
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_forecast_chart(predictions, current_pm25):
    """Create an interactive forecast chart."""
    hours = list(range(1, OUTPUT_HORIZON + 1))
    
    # Create figure
    fig = go.Figure()
    
    # Add current value marker
    fig.add_trace(go.Scatter(
        x=[0],
        y=[current_pm25],
        mode='markers',
        name='Current',
        marker=dict(size=15, color='#00d4ff', symbol='diamond'),
        hovertemplate='Current: %{y:.1f} μg/m³<extra></extra>'
    ))
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=hours,
        y=predictions,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff6b6b', width=3),
        marker=dict(size=10),
        hovertemplate='Hour +%{x}: %{y:.1f} μg/m³<extra></extra>'
    ))
    
    # Add AQI threshold lines
    fig.add_hline(y=35, line_dash="dash", line_color="#4CAF50", 
                  annotation_text="Good (35)", annotation_position="right")
    fig.add_hline(y=75, line_dash="dash", line_color="#FF9800",
                  annotation_text="Moderate (75)", annotation_position="right")
    fig.add_hline(y=150, line_dash="dash", line_color="#F44336",
                  annotation_text="Unhealthy (150)", annotation_position="right")
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='🌫️ PM2.5 12-Hour Forecast',
            font=dict(size=20, color='white')
        ),
        xaxis_title='Hours Ahead',
        yaxis_title='PM2.5 (μg/m³)',
        template='plotly_dark',
        height=450,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig


def create_gauge_chart(value, title="PM2.5 Level"):
    """Create a gauge chart for PM2.5 level."""
    category = get_aqi_category(value)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number={'suffix': ' μg/m³', 'font': {'size': 40, 'color': 'white'}},
        title={'text': title, 'font': {'size': 18, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 300], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': category['color']},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 12], 'color': 'rgba(76, 175, 80, 0.3)'},
                {'range': [12, 35.4], 'color': 'rgba(255, 235, 59, 0.3)'},
                {'range': [35.4, 55.4], 'color': 'rgba(255, 152, 0, 0.3)'},
                {'range': [55.4, 150.4], 'color': 'rgba(244, 67, 54, 0.3)'},
                {'range': [150.4, 250.4], 'color': 'rgba(156, 39, 176, 0.3)'},
                {'range': [250.4, 300], 'color': 'rgba(123, 31, 162, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        template='plotly_dark',
        height=300,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_feature_importance_chart(features_dict):
    """Create a radar chart showing feature values."""
    categories = list(features_dict.keys())
    values = list(features_dict.values())
    
    # Normalize values for radar chart (0-100 scale)
    normalized = []
    for feat, val in features_dict.items():
        info = FEATURE_INFO[feat]
        norm_val = ((val - info['min']) / (info['max'] - info['min'])) * 100
        normalized.append(min(100, max(0, norm_val)))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized + [normalized[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(0, 212, 255, 0.3)',
        line=dict(color='#00d4ff', width=2),
        name='Current Values'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color='white'),
                gridcolor='rgba(255,255,255,0.2)'
            ),
            angularaxis=dict(
                tickfont=dict(color='white', size=10),
                gridcolor='rgba(255,255,255,0.2)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        template='plotly_dark',
        height=400,
        margin=dict(l=80, r=80, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌫️ PM2.5 Air Quality Forecaster</h1>
        <p>Hybrid Transformer-LSTM Deep Learning Model for 12-Hour Air Quality Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, feature_scaler, target_scaler, model_loaded = load_model_and_scalers()
    
    if not model_loaded:
        st.error("❌ Failed to load the model. Please ensure model files are in the 'model' directory.")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Input Parameters")
        st.markdown("---")
        
        # Mode selection
        input_mode = st.radio(
            "Input Mode",
            ["🎛️ Manual Input", "📊 Sample Data", "📁 Upload CSV"],
            help="Choose how to provide input data"
        )
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        This application uses a **Hybrid Transformer-LSTM** 
        deep learning model to predict PM2.5 air quality 
        levels up to **12 hours ahead**.
        
        **Model Architecture:**
        - Transformer Encoder (4 attention heads)
        - LSTM Decoder (2 layers, 128 units)
        - Input: 48 hours of historical data
        - Output: 12-hour forecast
        
        **Features Used:**
        - PM2.5, PM10 (Particulate Matter)
        - Temperature, Pressure, Dew Point
        - Wind Speed
        - NO₂, SO₂, CO, O₃ (Pollutants)
        """)
        
        st.markdown("---")
        st.markdown("### 👨‍💻 Developer")
        st.markdown("**Sheryar Sher**")
        st.markdown("*Attention-Based Deep Learning for Hyper-Local Air Quality Prediction*")
    
    # Main content based on input mode
    if input_mode == "🎛️ Manual Input":
        st.markdown("## 📝 Enter Current Environmental Conditions")
        st.markdown("Provide the current readings from environmental sensors. The model will use these values to generate a 12-hour PM2.5 forecast.")
        
        # Create input columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🌡️ Weather Parameters")
            temp = st.slider("Temperature (°C)", -20.0, 45.0, 20.0, 0.5)
            pres = st.slider("Pressure (hPa)", 980.0, 1050.0, 1013.0, 1.0)
            dewp = st.slider("Dew Point (°C)", -40.0, 30.0, 10.0, 0.5)
            wspm = st.slider("Wind Speed (m/s)", 0.0, 20.0, 2.0, 0.1)
        
        with col2:
            st.markdown("### 🏭 Particulate Matter")
            pm25 = st.slider("Current PM2.5 (μg/m³)", 0.0, 500.0, 50.0, 1.0)
            pm10 = st.slider("PM10 (μg/m³)", 0.0, 600.0, 80.0, 1.0)
        
        with col3:
            st.markdown("### 💨 Gas Pollutants")
            no2 = st.slider("NO₂ (μg/m³)", 0.0, 300.0, 40.0, 1.0)
            so2 = st.slider("SO₂ (μg/m³)", 0.0, 200.0, 15.0, 1.0)
            co = st.slider("CO (μg/m³)", 0.0, 10000.0, 800.0, 10.0)
            o3 = st.slider("O₃ (μg/m³)", 0.0, 300.0, 50.0, 1.0)
        
        # Collect feature values
        current_features = {
            'PM2.5': pm25, 'PM10': pm10, 'TEMP': temp, 'PRES': pres,
            'DEWP': dewp, 'WSPM': wspm, 'NO2': no2, 'SO2': so2, 'CO': co, 'O3': o3
        }
        
        # Create input data (replicate current values for 48 hours as baseline)
        input_data = np.tile(list(current_features.values()), (INPUT_WINDOW, 1))
        
        # Predict button
        st.markdown("---")
        if st.button("🔮 Generate 12-Hour Forecast", type="primary", use_container_width=True):
            with st.spinner("🧠 Running Hybrid Transformer-LSTM Model..."):
                predictions = predict_pm25(model, feature_scaler, target_scaler, input_data)
            
            if predictions is not None:
                st.markdown("---")
                st.markdown("## 📊 Prediction Results")
                
                # Display current AQI
                current_cat = get_aqi_category(pm25)
                avg_prediction = np.mean(predictions)
                max_prediction = np.max(predictions)
                pred_cat = get_aqi_category(avg_prediction)
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="Current PM2.5",
                        value=f"{pm25:.1f} μg/m³",
                        delta=None
                    )
                    st.markdown(f"**Status:** {current_cat['emoji']} {current_cat['level']}")
                
                with col2:
                    delta = avg_prediction - pm25
                    st.metric(
                        label="Avg Forecast (12h)",
                        value=f"{avg_prediction:.1f} μg/m³",
                        delta=f"{delta:+.1f} μg/m³"
                    )
                
                with col3:
                    st.metric(
                        label="Peak Forecast",
                        value=f"{max_prediction:.1f} μg/m³",
                        delta=f"+{max_prediction - pm25:.1f} from current"
                    )
                
                with col4:
                    min_prediction = np.min(predictions)
                    st.metric(
                        label="Min Forecast",
                        value=f"{min_prediction:.1f} μg/m³",
                        delta=f"{min_prediction - pm25:+.1f} from current"
                    )
                
                # Forecast chart
                st.plotly_chart(create_forecast_chart(predictions, pm25), use_container_width=True)
                
                # Two column layout for gauge and radar
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Average Forecast Level")
                    st.plotly_chart(create_gauge_chart(avg_prediction, "12-Hour Average"), use_container_width=True)
                    
                    # Health advisory
                    st.markdown(f"""
                    <div class="prediction-box" style="border-color: {pred_cat['color']}">
                        <h3 style="color: {pred_cat['color']}">{pred_cat['emoji']} {pred_cat['level']}</h3>
                        <p style="color: #ccc">{pred_cat['advice']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 🎯 Input Feature Profile")
                    st.plotly_chart(create_feature_importance_chart(current_features), use_container_width=True)
                
                # Hourly breakdown table
                st.markdown("### 📋 Hourly Forecast Breakdown")
                forecast_df = pd.DataFrame({
                    'Hour': [f'+{i}h' for i in range(1, OUTPUT_HORIZON + 1)],
                    'PM2.5 (μg/m³)': [f"{p:.1f}" for p in predictions],
                    'Status': [get_aqi_category(p)['level'] for p in predictions],
                    'Advisory': [get_aqi_category(p)['emoji'] for p in predictions]
                })
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    
    elif input_mode == "📊 Sample Data":
        st.markdown("## 📊 Sample Scenarios")
        st.markdown("Select a pre-defined scenario to see the model's predictions.")
        
        scenarios = {
            "🌤️ Clean Day": {'PM2.5': 15, 'PM10': 25, 'TEMP': 22, 'PRES': 1015, 'DEWP': 12, 'WSPM': 4, 'NO2': 20, 'SO2': 5, 'CO': 400, 'O3': 80},
            "🌫️ Moderate Pollution": {'PM2.5': 60, 'PM10': 100, 'TEMP': 18, 'PRES': 1010, 'DEWP': 8, 'WSPM': 2, 'NO2': 60, 'SO2': 25, 'CO': 1200, 'O3': 40},
            "😷 Heavy Pollution": {'PM2.5': 180, 'PM10': 250, 'TEMP': 5, 'PRES': 1020, 'DEWP': -5, 'WSPM': 1, 'NO2': 120, 'SO2': 80, 'CO': 3000, 'O3': 20},
            "🚨 Severe Smog": {'PM2.5': 350, 'PM10': 450, 'TEMP': 2, 'PRES': 1025, 'DEWP': -10, 'WSPM': 0.5, 'NO2': 200, 'SO2': 150, 'CO': 6000, 'O3': 10}
        }
        
        selected_scenario = st.selectbox("Select Scenario", list(scenarios.keys()))
        current_features = scenarios[selected_scenario]
        
        # Display scenario details
        st.markdown("### Current Conditions")
        cols = st.columns(5)
        for i, (feat, val) in enumerate(current_features.items()):
            with cols[i % 5]:
                st.metric(FEATURE_INFO[feat]['name'], f"{val} {FEATURE_INFO[feat]['unit']}")
        
        # Create input data
        input_data = np.tile(list(current_features.values()), (INPUT_WINDOW, 1))
        
        if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
            with st.spinner("🧠 Running prediction..."):
                predictions = predict_pm25(model, feature_scaler, target_scaler, input_data)
            
            if predictions is not None:
                st.plotly_chart(create_forecast_chart(predictions, current_features['PM2.5']), use_container_width=True)
                
                avg_pred = np.mean(predictions)
                cat = get_aqi_category(avg_pred)
                st.markdown(f"""
                <div class="prediction-box" style="border-color: {cat['color']}">
                    <h2 style="color: {cat['color']}">{cat['emoji']} {cat['level']}</h2>
                    <h3 style="color: white">Average 12-Hour Forecast: {avg_pred:.1f} μg/m³</h3>
                    <p style="color: #ccc">{cat['advice']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    else:  # Upload CSV
        st.markdown("## 📁 Upload Historical Data")
        st.markdown("""
        Upload a CSV file with 48 hours of historical data. The file should contain the following columns:
        `PM2.5, PM10, TEMP, PRES, DEWP, WSPM, NO2, SO2, CO, O3`
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head(10))
                
                if len(df) >= INPUT_WINDOW:
                    # Use last 48 rows
                    input_data = df[FEATURE_COLUMNS].tail(INPUT_WINDOW).values
                    
                    if st.button("🔮 Generate Forecast", type="primary"):
                        with st.spinner("🧠 Running prediction..."):
                            predictions = predict_pm25(model, feature_scaler, target_scaler, input_data)
                        
                        if predictions is not None:
                            current_pm25 = input_data[-1, 0]
                            st.plotly_chart(create_forecast_chart(predictions, current_pm25), use_container_width=True)
                else:
                    st.warning(f"⚠️ Need at least {INPUT_WINDOW} rows of data. Your file has {len(df)} rows.")
            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🌫️ <strong>PM2.5 Air Quality Forecaster</strong> | Powered by Hybrid Transformer-LSTM Deep Learning</p>
        <p>Developed by <strong>Sheryar Sher</strong> | Attention-Based Deep Learning for Hyper-Local Air Quality Prediction</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

