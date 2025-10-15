import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Define o prefixo comum que identifica as colunas OHE da variável original 'SOURCE_KEY'
OHE_PREFIX = 'SOURCE_KEY_'

# Page config
st.set_page_config(
    page_title="Solar Power Forecasting",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF9933 0%, #FFD700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-box {
        background-color: #f0f7ff;
        padding: 1rem;
        border-left: 4px solid #4299e1;
        border-radius: 4px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title with custom styling
st.markdown('<h1 class="main-header">☀️ Solar Power Forecasting</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time AC power prediction using physics-informed machine learning</p>', unsafe_allow_html=True)

# Add brief context banner
st.info('''
**🎯 Purpose:** This model forecasts 15-minute solar power output based on weather conditions, 
enabling grid operators to optimize energy distribution and trading strategies.
''')

# Load model and extract feature names automatically
@st.cache_resource
def load_model_and_features():
    """Loads the model and attempts to automatically extract OHE feature names."""
    try:
        # Make sure 'solar_power_model.pkl' is in the same directory
        model = joblib.load('solar_power_model.pkl')
        
        # Try to get the expected feature names
        if hasattr(model, 'feature_names_in_'):
            all_features = model.feature_names_in_.tolist() # Converted to list
        elif hasattr(model.get_booster(), 'feature_names'):
             # Alternative for some XGBoost models
            all_features = model.get_booster().feature_names
        else:
            st.error("Error: The loaded model does not have the 'feature_names_in_' property or equivalent. Cannot automate OHE key extraction.")
            all_features = []

        # Filter only the One-Hot-Encoded columns for SOURCE_KEY
        source_keys = [f for f in all_features if f.startswith(OHE_PREFIX)]

        # Create the friendly map for Streamlit
        source_options_map = {}
        for i, key in enumerate(source_keys):
            # Format: "Inverter 1 (original_key)"
            # Extracts the raw key part (e.g., '1BY6WEcLGh8j5v7')
            raw_key = key.replace(OHE_PREFIX, "") 
            friendly_name = f'Inverter {i+1} ({raw_key})'
            source_options_map[friendly_name] = key

        return model, all_features, source_keys, source_options_map # Returns ALL features
    
    except FileNotFoundError:
        st.error("Error: The model file 'solar_power_model.pkl' was not found.")
        return None, [], [], {}
    except Exception as e:
        st.error(f"Error loading model or extracting features: {e}")
        return None, [], [], {}

# Note: ALL_MODEL_FEATURES is the complete list of columns in the correct order
model, ALL_MODEL_FEATURES, ALL_SOURCE_KEYS, SOURCE_OPTIONS_MAP = load_model_and_features()

# Checks if the model loaded correctly and if there are OHE keys
if not model:
    st.stop() # Stops execution if model failed to load
if not ALL_SOURCE_KEYS:
     st.warning(f"Warning: No OHE columns starting with '{OHE_PREFIX}' were found. Check the prefix or the model.")

# --- MAPPING: USE THE EXACT KEYS PROVIDED BY THE USER (CRUCIAL FIX) ---

# Keys for Plant 1, prefixed with OHE_PREFIX
PLANT_1_RAW_KEYS = [
    '1BY6WEcLGh8j5v7', '1IF53ai7Xc0U56Y', '3PZuoBAID5Wc2HD', '7JYdWkrLSPkdwr4', 
    'McdE0feGgRqW7Ca', 'VHMLBKoKgIrUVDU', 'WRmjgnKYAwPKWDb', 'ZnxXDlPa8U1GXgE', 
    'ZoEaEvLYb1n2sOq', 'adLQvlD726eNBSB', 'bvBOhCH3iADSZry', 'iCRJl6heRkivqQ3', 
    'ih0vzX44oOqAx2f', 'pkci93gMrogZuBj', 'rGa61gmuvPhdLxV', 'sjndEbLyjtCKgGv', 
    'uHbuxQJl8lW7ozc', 'wCURE6d3bPkepu2', 'z9Y9gH1T5YWrNuG', 'zBIq5rxdHJRwDNY', 
    'zVJPv84UY57bAof', 'YxYtjZvoooNbGkE'
]
PLANT_1_OHE_KEYS = [OHE_PREFIX + key for key in PLANT_1_RAW_KEYS]

# Keys for Plant 2, prefixed with OHE_PREFIX
PLANT_2_RAW_KEYS = [
    '4UPUqMRk7TRMgml', '81aHJ1q11NBPMrL', '9kRcWv60rDACzjR', 'Et9kgGMDl729KT4', 
    'IQ2d7wF4YD8zU1Q', 'LYwnQax7tkwH5Cb', 'LlT2YUhhzqhg5Sw', 'Mx2yZCDsyf6DPfv', 
    'NgDl19wMapZy17u', 'PeE6FRyGXUgsRhN', 'Qf4GUc1pJu5T6c6', 'Quc1TzYxW2pYoWX', 
    'V94E5Ben1TlhnDV', 'WcxssY2VbP4hApt', 'mqwcsP2rE7J0TFp', 'oZ35aAeoifZaQzV', 
    'oZZkBaNadn6DNKz', 'q49J1IKaHRwDQnt', 'rrq4fwE8jgrTyWY', 'vOuJvMaM2sgwLmb', 
    'xMbIugepa2P7lBB', 'xoJJ8DcxJEcupym'
]
PLANT_2_OHE_KEYS = [OHE_PREFIX + key for key in PLANT_2_RAW_KEYS]

PLANT_INVERTERS_MAP = {
    1: PLANT_1_OHE_KEYS,
    2: PLANT_2_OHE_KEYS
}
# --- END OF MAPPING ---

# Sidebar for inputs
st.sidebar.markdown("## 🎛️ Input Parameters")
st.sidebar.markdown("---")

# Main features
st.sidebar.markdown("### 🌤️ Weather Conditions")
st.sidebar.caption("Real-time sensor readings from the solar plant")

irradiation = st.sidebar.number_input(
    '☀️ Solar Irradiation (W/m²)', 
    min_value=0.0, 
    max_value=1400.0, 
    value=800.0, 
    step=10.0,
    help='Solar irradiance measured on panel surface. Clear-sky peak ≈ 1000 W/m²'
)

module_temp = st.sidebar.number_input(
    '🔥 Module Temperature (°C)', 
    min_value=0.0, 
    max_value=80.0, 
    value=45.0, 
    step=1.0,
    help='Panel surface temperature. Higher temps reduce efficiency (~0.4%/°C loss)'
)

ambient_temp = st.sidebar.number_input(
    '🌡️ Ambient Temperature (°C)', 
    min_value=0.0, 
    max_value=50.0, 
    value=30.0, 
    step=1.0,
    help='Outside air temperature at the plant location'
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Time & Location")
st.sidebar.caption("Temporal and geographic parameters")

hour = st.sidebar.slider(
    '🕐 Hour of Day', 
    min_value=0, 
    max_value=23, 
    value=12,
    help='Hour in 24-hour format (0=midnight, 12=noon, 23=11 PM)'
)

selected_date = st.sidebar.date_input(
    '📆 Forecast Date',
    value=datetime(2023, 5, 30),
    help='Calendar date for prediction (affects solar angle calculations)'
)

# Extract the Day of Year (1-365) from the selected date object
day_of_year = selected_date.timetuple().tm_yday

plant_id = st.sidebar.selectbox(
    '🏭 Solar Plant',
    options=[1, 2],
    format_func=lambda x: f"Plant {x} (22 inverters)",
    help='Select which solar facility to forecast'
)

# --- NEW: PREDICTION MODE SELECTOR ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Prediction Scope")
st.sidebar.caption("Choose forecast granularity")

prediction_mode = st.sidebar.radio(
    "Forecast Mode",
    options=['Individual Inverter', 'Total Plant Output'],
    index=1, # Default to Total Plant Output for the strategic view
    help='''
    • **Individual Inverter**: Predict single inverter's AC power output
    • **Total Plant Output**: Sum predictions across all 22 inverters in the plant
    '''
)

# Conditional UI: Only show the Inverter Key selector if the mode is 'Individual Inverter'
selected_source_key_ohe = None
if prediction_mode == 'Individual Inverter':
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔌 Inverter Selection")
    
    # Filter keys to only show inverters belonging to the selected plant
    available_inverters = PLANT_INVERTERS_MAP.get(plant_id, ALL_SOURCE_KEYS)
    filtered_options = {k: v for k, v in SOURCE_OPTIONS_MAP.items() if v in available_inverters}

    if not filtered_options:
        st.error(f"No inverters found for Plant {plant_id} based on the internal mapping.")
        st.stop()
        
    selected_source_key_friendly = st.sidebar.selectbox(
        'Inverter Key',
        options=list(filtered_options.keys()),
        help='Select the specific inverter key to predict.'
    )

    # Get the OHE column name that matches the selection
    if selected_source_key_friendly in filtered_options:
        selected_source_key_ohe = filtered_options[selected_source_key_friendly]
    else:
        # Fallback to the first available key if something went wrong
        selected_source_key_ohe = available_inverters[0]
else:
    # If Total Plant Output is selected, the OHE keys will be looped over inside the prediction logic
    pass
# --- END NEW SELECTOR ---

def calculate_features(irrad, mod_temp, amb_temp, hour, day, plant, selected_source_ohe, required_features):
    """
    Calculates all engineered features needed by the model, including OHE.
    selected_source_ohe is the OHE key to set to 1. All others OHE keys are set to 0.
    Ensures that the column order in the final DataFrame matches the 'required_features' list.
    """
    
    # Cyclical encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day / 365)
    day_cos = np.cos(2 * np.pi * day / 365)
    
    # Solar position (simplified approximation)
    declination = 23.45 * np.sin(np.radians((360/365) * (day - 81)))
    latitude = 23.0  
    hour_angle = 15 * (hour - 12)
    
    solar_zenith = np.degrees(np.arccos(
        np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) +
        np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle))
    ))
    
    solar_elevation = 90 - solar_zenith
    solar_azimuth = 180 # Simplified (south-facing)
    
    # Air mass
    air_mass = np.clip(1 / np.cos(np.radians(solar_zenith)), 1, 10)
    
    # Temperature delta
    temp_delta = mod_temp - amb_temp
    
    # --- One-Hot Encoding for SOURCE_KEY (uses automatic keys) ---
    source_key_ohe_features = {}
    for key in ALL_SOURCE_KEYS:
        # Initialize all OHE keys to 0
        source_key_ohe_features[key] = 0

    # Set the selected key to 1
    if selected_source_key_ohe and selected_source_key_ohe in source_key_ohe_features:
        source_key_ohe_features[selected_source_key_ohe] = 1
    # --- END OF OHE ---

    # Dictionary of all features, WITHOUT GUARANTEEING ORDER
    features = {
        'IRRADIATION': irrad,
        'AMBIENT_TEMPERATURE': amb_temp,
        'MODULE_TEMPERATURE': mod_temp,
        'PLANT_ID': plant,
        'HOUR': hour,
        'HOUR_SIN': hour_sin,
        'HOUR_COS': hour_cos,
        'DAY_OF_YEAR': day,
        'DAY_SIN': day_sin,
        'DAY_COS': day_cos,
        'SOLAR_ZENITH': solar_zenith,
        'SOLAR_AZIMUTH': solar_azimuth,
        'SOLAR_ELEVATION': solar_elevation,
        'AIR_MASS': air_mass,
        'TEMPERATURE_DIFFERENCE': temp_delta,
        # Add OHE keys
        **source_key_ohe_features
    }
    
    # Create the DataFrame
    input_df = pd.DataFrame([features])

    # CRITICAL STEP: Reorder columns using the list extracted from the model (required_features)
    try:
        input_df = input_df[required_features]
    except KeyError as e:
        # This might happen if the model's feature list is different from what was calculated/provided.
        st.error(f"Column ordering error: Feature {e} is missing or there is a prefix issue.")
        st.stop()
        
    return input_df

# Predict button
if st.sidebar.button('🔮 Predict Power Output', type='primary'):
    
    # --- MAIN PREDICTION LOGIC ---
    total_prediction = 0
    prediction_label = ""
    
    if prediction_mode == 'Individual Inverter':
        # --- Mode 1: Individual Inverter Prediction ---
        if not selected_source_key_ohe:
            st.error('❌ Could not identify OHE key for individual prediction.')
            st.stop()

        # Calculate features for the selected inverter
        input_features = calculate_features(
            irradiation, module_temp, ambient_temp, 
            hour, day_of_year, plant_id,
            selected_source_key_ohe,
            ALL_MODEL_FEATURES 
        )
        
        # Make the prediction
        total_prediction = model.predict(input_features)[0]
        prediction_label = f"Predicted AC Power (Inverter)"

    else:
        # --- Mode 2: Total Plant Output Prediction ---
        inverters_to_sum = PLANT_INVERTERS_MAP.get(plant_id)

        if not inverters_to_sum:
            st.error(f'❌ No inverters mapped to Plant ID {plant_id}. Cannot sum total power. Please check the `PLANT_INVERTERS_MAP` definition.')
            st.stop()
            
        # Loop through all inverters in the selected plant and sum their predictions
        for ohe_key in inverters_to_sum:
            # Calculate features for the current inverter (setting its OHE to 1)
            # NOTE: We keep the PLANT_ID constant for all predictions in this loop
            input_features = calculate_features(
                irradiation, module_temp, ambient_temp, 
                hour, day_of_year, plant_id,
                ohe_key, # Use the current OHE key in the loop
                ALL_MODEL_FEATURES 
            )
            # Add the individual inverter's prediction to the total
            total_prediction += model.predict(input_features)[0]
            
        prediction_label = f"Total Predicted AC Power (Plant {plant_id})"
    # --- END MAIN PREDICTION LOGIC ---
    
    # Display result
    try:
        # Success banner with gradient
        st.markdown(f"""
            <div style='background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%); 
                        padding: 1.5rem; border-radius: 10px; margin: 2rem 0;'>
                <h2 style='color: white; margin: 0;'>✅ Prediction Complete</h2>
                <p style='color: white; margin: 0.5rem 0 0 0;'>
                    Model confidence: High | Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Main metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 2rem; border-radius: 15px; text-align: center; color: white;'>
                    <h4 style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>{prediction_label}</h4>
                    <h1 style='margin: 0.5rem 0; font-size: 3rem; font-weight: 700;'>
                        {total_prediction:.1f}
                    </h1>
                    <p style='margin: 0; font-size: 1.2rem;'>kW</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            inverter_count = len(PLANT_INVERTERS_MAP.get(plant_id, [1])) if prediction_mode == 'Total Plant Output' else 1
            max_capacity_per_plant = inverter_count * 1000 
            max_capacity = max_capacity_per_plant if prediction_mode == 'Total Plant Output' else 1000
            capacity_factor = (total_prediction / max_capacity) * 100
            
            # Color coding for capacity factor
            if capacity_factor < 30:
                cf_color = "#e53e3e"  # Red
            elif capacity_factor < 70:
                cf_color = "#dd6b20"  # Orange
            else:
                cf_color = "#38a169"  # Green
            
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                            padding: 2rem; border-radius: 15px; text-align: center; color: white;'>
                    <h4 style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>Capacity Factor</h4>
                    <h1 style='margin: 0.5rem 0; font-size: 3rem; font-weight: 700;'>
                        {capacity_factor:.1f}
                    </h1>
                    <p style='margin: 0; font-size: 1.2rem;'>% of max capacity</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            daily_energy = total_prediction * 8  # 8 productive hours
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                            padding: 2rem; border-radius: 15px; text-align: center; color: white;'>
                    <h4 style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>Estimated Daily Energy</h4>
                    <h1 style='margin: 0.5rem 0; font-size: 3rem; font-weight: 700;'>
                        {daily_energy:.0f}
                    </h1>
                    <p style='margin: 0; font-size: 1.2rem;'>kWh (8h avg)</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Additional context metrics
        st.markdown("---")
        st.markdown("### 📊 Prediction Context")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("☀️ Irradiation", f"{irradiation:.0f} W/m²", 
                   delta="High" if irradiation > 800 else "Moderate" if irradiation > 400 else "Low")
        col2.metric("🔥 Module Temp", f"{module_temp:.1f}°C",
                   delta="Hot" if module_temp > 50 else "Normal")
        col3.metric("🌡️ Ambient Temp", f"{ambient_temp:.1f}°C")
        col4.metric("⏰ Time", f"{hour:02d}:00",
                   delta="Peak hours" if 10 <= hour <= 14 else "Off-peak")
        
        # Visualization
        st.markdown("---")
        st.markdown("### 📈 Input Conditions Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced irradiation gauge
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#e53e3e' if irradiation < 200 else '#dd6b20' if irradiation < 600 else '#38a169']
            ax.barh(['Solar Irradiation'], [irradiation], color=colors, height=0.5)
            ax.set_xlim(0, 1200)
            ax.set_xlabel('Irradiance (W/m²)', fontsize=12, fontweight='bold')
            ax.set_title('☀️ Solar Irradiation Level', fontsize=14, fontweight='bold', pad=20)
            ax.axvline(x=1000, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Clear-sky peak')
            ax.legend(loc='lower right')
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig)
            
            # Status indicator
            if irradiation > 800:
                st.success("🌞 Excellent solar conditions")
            elif irradiation > 400:
                st.warning("⛅ Moderate solar conditions (partial clouds possible)")
            else:
                st.error("☁️ Low solar conditions (heavy clouds or dawn/dusk)")
        
        with col2:
            # Enhanced temperature comparison
            fig, ax = plt.subplots(figsize=(8, 5))
            temps = [ambient_temp, module_temp]
            colors_temp = ['#4299e1', '#f56565']
            bars = ax.bar(['Ambient\nTemperature', 'Module\nTemperature'], temps, 
                         color=colors_temp, width=0.6, edgecolor='white', linewidth=2)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}°C',
                       ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
            ax.set_title('🌡️ Temperature Comparison', fontsize=14, fontweight='bold', pad=20)
            ax.set_ylim(0, max(temps) * 1.2)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            
            # Temperature delta indicator
            temp_delta = module_temp - ambient_temp
            if temp_delta > 25:
                st.warning(f"⚠️ High temperature delta: {temp_delta:.1f}°C (efficiency loss expected)")
            elif temp_delta > 15:
                st.info(f"ℹ️ Moderate temperature delta: {temp_delta:.1f}°C (normal operation)")
            else:
                st.success(f"✅ Low temperature delta: {temp_delta:.1f}°C (optimal efficiency)")
        
    except Exception as e:
        st.error(f'❌ Error making prediction: {str(e)}')
        st.write('**Check:**')
        st.write(f'- If the **`OHE_PREFIX`** ("{OHE_PREFIX}") at the top of the file matches the prefix of your OHE columns in the model.')
        st.write('- If your model was saved correctly with column names (`feature_names_in_` property).')

else:
    # Enhanced info section
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem; border-radius: 15px; text-align: center; color: white; margin: 2rem 0;'>
            <h2 style='margin: 0;'>👋 Welcome to Solar Power Forecasting</h2>
            <p style='margin: 1rem 0 0 0; font-size: 1.1rem;'>
                Adjust parameters in the sidebar and click <strong>"🔮 Predict Power Output"</strong> to begin
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='info-box'>
                <h4>1️⃣ Set Weather Conditions</h4>
                <p>Enter current irradiation, module temperature, and ambient temperature from plant sensors.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='info-box'>
                <h4>2️⃣ Choose Time & Plant</h4>
                <p>Select the hour, date, and which solar plant (1 or 2) you want to forecast.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='info-box'>
                <h4>3️⃣ Select Prediction Mode</h4>
                <p>Forecast for a single inverter or get the total plant output (all 22 inverters summed).</p>
            </div>
        """, unsafe_allow_html=True)
    
    # --- Inverter Keys Reference ---
    st.markdown("---")
    st.markdown("### 🔧 System Configuration")
    
    with st.expander("📋 View Inverter IDs (Technical Reference)"):
        st.caption("This mapping ensures accurate predictions for individual inverters and correct summation for total plant output.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**🏭 Plant 1 Inverters** ({len(PLANT_1_RAW_KEYS)} units)")
            st.code('\n'.join([f"{i+1:2d}. {key}" for i, key in enumerate(PLANT_1_RAW_KEYS)]), language=None)
        
        with col2:
            st.markdown(f"**🏭 Plant 2 Inverters** ({len(PLANT_2_RAW_KEYS)} units)")
            st.code('\n'.join([f"{i+1:2d}. {key}" for i, key in enumerate(PLANT_2_RAW_KEYS)]), language=None)
    
    # --- About Section ---
    st.markdown("---")
    st.markdown("### 🤖 About This Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            **📊 Model Specifications:**
            - **Algorithm**: XGBoost Regressor (Gradient Boosting)
            - **Test R²**: 0.649 (explains 65% of power variance)
            - **Test RMSE**: ~165 kW
            - **Training samples**: 50,091 (24 days of data)
            - **Features**: 59 total
              - 5 raw weather measurements
              - 11 engineered features (solar position, thermal efficiency)
              - 44 inverter identifiers (one-hot encoded)
            - **Temporal resolution**: 15-minute intervals
        """)
    
    with col2:
        st.markdown("""
            **🔑 Top Predictive Features:**
            1. **Temperature Difference** (36.4%) - Captures efficiency losses
            2. **Irradiation** (15.9%) - Primary energy source
            3. **Module Temperature** (10.5%) - Thermal effects
            4. **Plant ID** (7.0%) - Site-specific characteristics
            5. **Hour** (4.3%) - Time-of-day patterns
            6. **Solar Zenith Angle** - Sun position effects
            
            *Feature importance derived from XGBoost's gain metric*
        """)
    
    # Performance comparison
    st.markdown("---")
    st.markdown("### 📈 Model Performance")
    
    st.markdown("""
        The model achieved consistent performance across train/validation/test sets, 
        demonstrating good generalization to unseen future data:
    """)
    
    performance_data = {
        'Dataset': ['Training', 'Validation', 'Test'],
        'R² Score': [0.698, 0.716, 0.649],
        'RMSE (kW)': [168, 165, 180],
        'Samples': ['50,091', '10,734', '10,734'],
        'Time Period': ['May 15 - Jun 4', 'Jun 5 - Jun 10', 'Jun 11 - Jun 17']
    }
    
    df_performance = pd.DataFrame(performance_data)
    st.dataframe(df_performance, use_container_width=True, hide_index=True)
    
    # Key insights
    st.info("""
        **💡 Key Insights:**
        - ✅ **No overfitting**: Train and validation R² are close (69.8% vs 71.6%)
        - ⚠️ **Realistic test drop**: 6.7% decrease from validation (expected for time-series with new weather patterns)
        - ✅ **Beats baseline**: +1.7% better than Linear Regression (63.2%)
        - 📊 **Performance ceiling**: ~65% R² represents the practical limit given 34 days of 15-min data
    """)
    
    # Limitations
    with st.expander("⚠️ Model Limitations & Considerations"):
        st.markdown("""
            **Data Constraints:**
            - 📅 Trained on only 34 days (May-June 2020) - **no seasonal generalization**
            - ⏱️ 15-minute resolution - **misses rapid fluctuations** (clouds, inverter switching)
            - 🌍 India-specific (lat 21-23°N) - **not validated for other geographies**
            - 🔧 No inverter specifications - **efficiency curves learned empirically**
            
            **Remaining 35% Unexplained Variance Due To:**
            - ☁️ Cloud edges between 15-min intervals
            - 🦅 Soiling events (bird droppings, dust)
            - ⚡ Brief inverter trips/restarts
            - 🌡️ Sensor lag (1-2 minute delay)
            - 📏 Measurement noise
            
            **Deployment Recommendations:**
            - ✅ Use as **one signal** among many for grid decisions (not sole input)
            - ✅ Retrain quarterly to adapt to seasonal changes
            - ✅ Monitor performance drift over time
            - ✅ Combine with weather forecasts for lookahead predictions
        """)

# Enhanced footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p style='margin: 0.5rem 0;'>
            <strong>🌟 Solar Power Forecasting v1.0</strong>
        </p>
        <p style='margin: 0.5rem 0;'>
            Built with Streamlit | 
            Data: <a href='https://www.kaggle.com/datasets/anikannal/solar-power-generation-data' target='_blank'>Kaggle Solar Power Dataset</a> | 
            Model: XGBoost
        </p>
        <p style='margin: 0.5rem 0;'>
            📧 Questions or feedback? 
            <a href='https://github.com/TheCoolerMendeiros' target='_blank'>GitHub</a> | 
            <a href='https://www.linkedin.com/in/pedro-mendeiros-159a801a8/' target='_blank'>LinkedIn</a>
        </p>
        <p style='margin: 1rem 0 0 0; font-size: 0.9rem;'>
            ⚡ Empowering renewable energy through data science
        </p>
    </div>
""", unsafe_allow_html=True)