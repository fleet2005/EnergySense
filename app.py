import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
import json
import warnings
import folium
from streamlit_folium import folium_static 
# Suppress warnings
warnings.filterwarnings('ignore')
# Import the classes from the prediction module
from prediction import (
    RenewableEnergyModels,
    RenewableEnergyForecaster,
    fetch_forecast_data,
    generate_forecast_report
)

# Define the available districts
districts = {
    'Chennai': {'lat': 13.0827, 'lon': 80.2707, 'coastal': True},
    'Coimbatore': {'lat': 11.0168, 'lon': 76.9558, 'coastal': False},
    'Madurai': {'lat': 9.9252, 'lon': 78.1198, 'coastal': False},
    'Trichy': {'lat': 10.7905, 'lon': 78.7047, 'coastal': False},
    'Salem': {'lat': 11.6643, 'lon': 78.1460, 'coastal': False},
    'Vellore': {'lat': 12.9165, 'lon': 79.1325, 'coastal': False},
    'Kanyakumari': {'lat': 8.0883, 'lon': 77.5385, 'coastal': True},
    'Tuticorin': {'lat': 8.7642, 'lon': 78.1348, 'coastal': True}
}



    


# Function to load models
@st.cache_resource
def load_models(model_dir='./saved_models'):
    try:
        models = RenewableEnergyModels(model_dir)
        if models.wind_model is None or models.solar_model is None:
            st.error("Failed to load required models. This is likely due to a scikit-learn version compatibility issue.")
            st.error("Please retrain the models by running: `python model.py` in your terminal.")
            st.info("The saved models were created with a different version of scikit-learn than what's currently installed.")
            return None
        return models
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None

# Function to generate forecasts
def generate_forecasts(selected_districts, forecast_days, models):
    forecaster = RenewableEnergyForecaster(models)
    district_forecasts = {}
    progress_bar = st.progress(0)
    
    for i, name in enumerate(selected_districts):
        progress_bar.progress((i) / len(selected_districts))
        status_text = st.empty()
        status_text.text(f"Generating forecast for {name}...")
        
        info = districts[name]
        forecast = forecaster.forecast_district_energy(name, info, forecast_days)
        
        if forecast:
            district_forecasts[name] = forecast
        
        progress_bar.progress((i + 1) / len(selected_districts))
    
    progress_bar.empty()
    return district_forecasts

# Function to create daily forecast plot with Plotly
def plot_daily_forecast(district_forecast, district_name):
    daily = district_forecast['daily_forecast']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily.index, 
        y=daily['wind_energy'], 
        mode='lines+markers',
        name='Wind Energy',
        marker=dict(size=8),
        line=dict(width=2, color='skyblue')
    ))
    
    fig.add_trace(go.Scatter(
        x=daily.index, 
        y=daily['solar_energy'], 
        mode='lines+markers',
        name='Solar Energy',
        marker=dict(size=8),
        line=dict(width=2, color='orange')
    ))
    
    if district_forecast['total_potential']['ocean'] > 0:
        fig.add_trace(go.Scatter(
            x=daily.index, 
            y=daily['ocean_energy'], 
            mode='lines+markers',
            name='Ocean Energy',
            marker=dict(size=8),
            line=dict(width=2, color='teal')
        ))
    
    fig.add_trace(go.Scatter(
        x=daily.index, 
        y=daily['total_energy'], 
        mode='lines+markers',
        name='Total Energy',
        marker=dict(size=10),
        line=dict(width=3, color='black', dash='dash')
    ))
    
    fig.update_layout(
        title=f'7-Day Renewable Energy Forecast for {district_name}',
        xaxis_title='Date',
        yaxis_title='Energy Potential (wh/m²)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    return fig

# Function to create hourly pattern plot with Plotly
def plot_hourly_pattern(district_forecast, district_name):
    hourly_pattern = district_forecast['hourly_patterns']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hourly_pattern.index, 
        y=hourly_pattern['wind_energy'], 
        mode='lines',
        name='Wind Energy',
        line=dict(width=2, color='skyblue')
    ))
    
    fig.add_trace(go.Scatter(
        x=hourly_pattern.index, 
        y=hourly_pattern['solar_energy'], 
        mode='lines',
        name='Solar Energy',
        line=dict(width=2, color='orange')
    ))
    
    if district_forecast['total_potential']['ocean'] > 0:
        fig.add_trace(go.Scatter(
            x=hourly_pattern.index, 
            y=hourly_pattern['ocean_energy'], 
            mode='lines',
            name='Ocean Energy',
            line=dict(width=2, color='teal')
        ))
    
    fig.add_trace(go.Scatter(
        x=hourly_pattern.index, 
        y=hourly_pattern['total_energy'], 
        mode='lines',
        name='Total Energy',
        line=dict(width=3, color='black', dash='dash')
    ))
    
    fig.update_layout(
        title=f'Average Hourly Energy Generation Pattern for {district_name}',
        xaxis_title='Hour of Day',
        yaxis_title='Energy Potential (wh/m²)',
        xaxis = dict(
            tickmode = 'array',
            tickvals = list(range(0, 24, 2))
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    return fig

# Function to create district comparison bar chart
def plot_district_comparison(district_forecasts):
    districts = list(district_forecasts.keys())
    wind_values = [df['total_potential']['wind'] for df in district_forecasts.values()]
    solar_values = [df['total_potential']['solar'] for df in district_forecasts.values()]
    ocean_values = [df['total_potential']['ocean'] for df in district_forecasts.values()]
    
    fig = go.Figure(data=[
        go.Bar(name='Wind', x=districts, y=wind_values, marker_color='skyblue'),
        go.Bar(name='Solar', x=districts, y=solar_values, marker_color='orange'),
        go.Bar(name='Ocean', x=districts, y=ocean_values, marker_color='teal')
    ])
    
    fig.update_layout(
        title='Forecasted Renewable Energy Potential by District',
        xaxis_title='District',
        yaxis_title='Average Daily Energy Potential (wh/m²)',
        barmode='group',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Function to create energy mix pie chart
def plot_energy_mix(report):
    labels = ['Wind', 'Solar', 'Ocean']
    values = [
        report['state_summary']['energy_mix']['wind'],
        report['state_summary']['energy_mix']['solar'],
        report['state_summary']['energy_mix']['ocean']
    ]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker_colors=['skyblue', 'orange', 'teal']
    )])
    
    fig.update_layout(
        title='Forecasted Energy Mix (%)',
        template='plotly_white'
    )
    
    return fig
# Set page configuration
st.set_page_config(
    page_title="EnergySense",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e7d32;
        margin-bottom: 0.5rem;
    }
    .section {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Fix navigation button issues */
    .stRadio > div > label > div[data-testid="stMarkdownContainer"] {
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    
    .stRadio > div > label {
        cursor: pointer !important;
        transition: none !important;
    }
    
    .stRadio > div > label:hover {
        background-color: #f0f0f0;
        border-radius: 0.25rem;
    }
    
    /* Ensure stable button behavior */
    .stRadio > div {
        margin-bottom: 0.5rem;
    }
    
    /* Prevent button overlap issues */
    .stSidebar .stRadio {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load sample data (replace with real data sources)
energy_data = pd.DataFrame({
    "Timestamp": pd.date_range(start="2023-10-01", periods=24, freq="H"),
    "Solar_Generation": np.random.uniform(0, 100, 24),
    "Wind_Generation": np.random.uniform(0, 50, 24),
    "Energy_Demand": np.random.uniform(50, 150, 24),
    "Battery_Storage": np.random.uniform(0, 100, 24),
})



# Navigation
st.sidebar.markdown("# EnergySense")
st.sidebar.markdown("---")

# Create a more stable navigation system
st.sidebar.markdown("### 🏠 Navigation")
page = st.sidebar.radio(
    "Select a service:",
    ["Home", "Renewable Energy Forecaster", "Grid Energy Allocation", "Turbine Thermal Defect Detection", "Power Map"],
    index=0,
    key="main_navigation"
)

# Add a visual indicator of current page
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current Page:** {page}")
st.sidebar.markdown("---")

# Home Page
if page == "Home":
    st.markdown("<h1 class='main-header'>EnergySense- Renewable Energy Management System</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to EnergySense
        
        This platform helps you monitor, optimize, and control renewable energy systems (solar, wind, etc.) 
        for efficient energy generation, storage, and distribution.
        
        **Key Features:**
        - 📊 Renewable Energy Forecaster
        - 🔮 Energy generation forecasting
        - 🔍 Turbine Thermal Defect Detection
        - 🗺️ Grid Energy Allocation
        
        Select a service from the sidebar to begin.
        """)
        
    with col2:
        st.image(r"images_frontend/a.jpg", use_container_width=True)
        st.image(r"images_frontend/b.jpg", use_container_width=True)
        



# Energy Forecasting Page
elif page == "Energy Forecasting":
    st.markdown("<h1 class='main-header'>Energy Forecasting</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Predict future energy generation based on weather forecasts and historical data.
    """)
    
    # Input for forecasting
    with st.form("forecasting_form"):
        location = st.text_input("Enter Location", "New York")
        days = st.slider("Forecast Period (Days)", 1, 7, 3)
        submitted = st.form_submit_button("Generate Forecast")
    
    if submitted:
        with st.spinner("Generating forecast..."):
            # Simulate forecasting (replace with actual model)
            forecast_data = pd.DataFrame({
                "Timestamp": pd.date_range(start=datetime.now(), periods=days * 24, freq="H"),
                "Solar_Forecast": np.random.uniform(0, 100, days * 24),
                "Wind_Forecast": np.random.uniform(0, 50, days * 24),
            })
            st.session_state.forecast_data = forecast_data
    
    if 'forecast_data' in st.session_state:
        st.markdown("<h3 class='sub-header'>Energy Generation Forecast</h3>", unsafe_allow_html=True)
        fig = px.line(st.session_state.forecast_data, x="Timestamp", y=["Solar_Forecast", "Wind_Forecast"],
                      labels={"value": "Energy (wh)", "variable": "Type"})
        st.plotly_chart(fig, use_container_width=True)


# Energy Allocation Page
elif page == "Grid Energy Allocation":
    st.markdown("<h1 class='main-header'>Grid Energy Allocation</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Optimize energy allocation to cities based on demand, transmission losses, and constraints.
    """)
    
    # List of cities in Tamil Nadu
    cities = ["Chennai", "Coimbatore", "Madurai", "Salem", "Thoothukudi", "Dindigul", "Nagapattinam", "Ramanathapuram"]

    # Base city demand data (in kWh) - will be modified by time of day
    base_city_demand = {
        "Chennai": {"residential": 1100, "commercial": 825, "industrial": 825},
        "Coimbatore": {"residential": 600, "commercial": 450, "industrial": 450},
        "Madurai": {"residential": 400, "commercial": 300, "industrial": 300},
        "Salem": {"residential": 340, "commercial": 255, "industrial": 255},
        "Thoothukudi": {"residential": 300, "commercial": 225, "industrial": 225},
        "Dindigul": {"residential": 280, "commercial": 210, "industrial": 210},
        "Nagapattinam": {"residential": 200, "commercial": 150, "industrial": 150},
        "Ramanathapuram": {"residential": 180, "commercial": 135, "industrial": 135},
    }
    
    # Use static base demand without time-based variation for simplicity
    city_demand = {}
    for city, base_demand in base_city_demand.items():
        city_demand[city] = {
            "residential": int(base_demand["residential"]),
            "commercial": int(base_demand["commercial"]),
            "industrial": int(base_demand["industrial"])
        }

    # Transmission losses (as a percentage of energy sent to each city)
    transmission_losses = {
        "Chennai": 0.05,
        "Coimbatore": 0.04,
        "Madurai": 0.05,
        "Salem": 0.06,
        "Thoothukudi": 0.06,
        "Dindigul": 0.07,
        "Nagapattinam": 0.08,
        "Ramanathapuram": 0.07,
    }

    # Transmission line capacity (maximum energy that can be transmitted to each city)
    transmission_capacity = {
        "Chennai": 3000,
        "Coimbatore": 2000,
        "Madurai": 1500,
        "Salem": 1200,
        "Thoothukudi": 1000,
        "Dindigul": 900,
        "Nagapattinam": 800,
        "Ramanathapuram": 700,
    }

    # Cost of energy distribution (per kWh)
    distribution_costs = {
        "Chennai": 0.10,
        "Coimbatore": 0.12,
        "Madurai": 0.11,
        "Salem": 0.13,
        "Thoothukudi": 0.14,
        "Dindigul": 0.15,
        "Nagapattinam": 0.16,
        "Ramanathapuram": 0.17,
    }

    # Geographical data (latitude and longitude)
    geo_data = {
        "Chennai": {"lat": 13.0827, "lon": 80.2707},
        "Coimbatore": {"lat": 11.0168, "lon": 76.9558},
        "Madurai": {"lat": 9.9252, "lon": 78.1198},
        "Salem": {"lat": 11.6643, "lon": 78.1460},
        "Thoothukudi": {"lat": 8.7642, "lon": 78.1348},
        "Dindigul": {"lat": 10.3621, "lon": 77.9765},
        "Nagapattinam": {"lat": 10.7667, "lon": 79.8417},
        "Ramanathapuram": {"lat": 9.3716, "lon": 78.8307},
    }

    # User input for grid parameters (simplified)
    st.sidebar.header("Grid Parameters Input")
    total_energy_available = st.sidebar.number_input("Total Available Energy (kWh)", min_value=0, value=8000)

    # Display total energy available
    st.sidebar.write(f"Total Energy Available: {total_energy_available:.2f} kWh")
    
    # Display current demand overview
    st.subheader("📊 Current Demand Overview")
    col1, col2, col3 = st.columns(3)
    
    total_residential = sum(city_demand[city]["residential"] for city in cities)
    total_commercial = sum(city_demand[city]["commercial"] for city in cities)
    total_industrial = sum(city_demand[city]["industrial"] for city in cities)
    total_demand = total_residential + total_commercial + total_industrial
    
    with col1:
        st.metric("Residential Demand", f"{total_residential:,} kWh")
    with col2:
        st.metric("Commercial Demand", f"{total_commercial:,} kWh")
    with col3:
        st.metric("Industrial Demand", f"{total_industrial:,} kWh")
    
    st.metric("Total Grid Demand", f"{total_demand:,} kWh", 
              f"Supply: {total_energy_available/total_demand*100:.1f}%" if total_demand > 0 else "N/A")
    
    # Show demand by city
    st.subheader("🏙️ Demand by City")
    demand_df = pd.DataFrame([
        {
            "City": city,
            "Residential (kWh)": city_demand[city]["residential"],
            "Commercial (kWh)": city_demand[city]["commercial"],
            "Industrial (kWh)": city_demand[city]["industrial"],
            "Total (kWh)": sum(city_demand[city].values())
        }
        for city in cities
    ])
    st.dataframe(demand_df, use_container_width=True)

    # Validation checks
    if total_energy_available <= 0:
        st.error("❌ Total energy available must be greater than 0")
        st.stop()

    # Button to run simplified allocation (greedy transportation)
    if st.sidebar.button("🚀 Allocate Energy to Cities"):
        remaining_supply = float(total_energy_available)
        per_city_demand_remaining = {
            city: float(sum(city_demand[city].values())) for city in cities
        }
        allocated_sent_per_city = {city: 0.0 for city in cities}

        # Effective cost per received kWh: distribution cost + loss penalty
        city_costs = {
            city: distribution_costs[city] + transmission_losses[city]
            for city in cities
        }

        # 1) Fairness seed pass: give each city an equal share of a small seed pool
        cities_needing = [c for c in cities if per_city_demand_remaining[c] > 0 and transmission_capacity[c] > 0]
        if remaining_supply > 0 and len(cities_needing) > 0:
            seed_fraction = 0.4  # allocate at most 40% of total supply in the seed
            seed_pool = min(remaining_supply, seed_fraction * float(total_energy_available))
            per_city_quota = seed_pool / len(cities_needing)
            for city in cities_needing:
                if remaining_supply <= 0:
                    break
                loss = transmission_losses[city]
                demand_needed = per_city_demand_remaining[city]
                if demand_needed <= 0:
                    continue
                send_needed = demand_needed / max(1e-9, (1 - loss))
                cap_left = max(0.0, transmission_capacity[city] - allocated_sent_per_city[city])
                to_send = min(per_city_quota, send_needed, cap_left, remaining_supply)
                if to_send > 0:
                    received = to_send * (1 - loss)
                    per_city_demand_remaining[city] = max(0.0, demand_needed - received)
                    allocated_sent_per_city[city] += to_send
                    remaining_supply -= to_send

        # 2) Strict-priority greedy: keep filling the cheapest city to completion
        sorted_cities = sorted(cities, key=lambda c: city_costs[c])

        while remaining_supply > 0:
            made_progress = False
            for city in sorted_cities:
                if remaining_supply <= 0:
                    break
                loss = transmission_losses[city]
                demand_needed = per_city_demand_remaining[city]
                if demand_needed <= 0:
                    continue
                send_needed = demand_needed / max(1e-9, (1 - loss))
                cap_left = max(0.0, transmission_capacity[city] - allocated_sent_per_city[city])
                to_send = min(send_needed, cap_left, remaining_supply)
                if to_send > 0:
                    received = to_send * (1 - loss)
                    per_city_demand_remaining[city] = max(0.0, demand_needed - received)
                    allocated_sent_per_city[city] += to_send
                    remaining_supply -= to_send
                    made_progress = True
            if not made_progress:
                break

        # Build results
        results = []
        for city in cities:
            allocated = float(allocated_sent_per_city[city])
            received = allocated * (1 - transmission_losses[city])
            unmet = max(0.0, float(sum(city_demand[city].values())) - received)
            results.append({
                "City": city,
                "Residential_Demand (kWh)": city_demand[city]["residential"],
                "Commercial_Demand (kWh)": city_demand[city]["commercial"],
                "Industrial_Demand (kWh)": city_demand[city]["industrial"],
                "Total_Demand (kWh)": float(sum(city_demand[city].values())),
                "Allocated_Energy (kWh)": float(allocated),
                "Transmission_Loss (kWh)": float(allocated * transmission_losses[city]),
                "Received_Energy (kWh)": float(received),
                "Unmet_Demand (kWh)": float(unmet),
                "Effective_Cost": float(city_costs[city])
            })

        allocation_df = pd.DataFrame(results)

        st.subheader("⚡ Energy Allocation Results (Greedy Transportation)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Allocated", f"{allocation_df['Allocated_Energy (kWh)'].sum():.0f} kWh")
        with col2:
            st.metric("Total Losses", f"{allocation_df['Transmission_Loss (kWh)'].sum():.0f} kWh")
        with col3:
            st.metric("Total Received", f"{allocation_df['Received_Energy (kWh)'].sum():.0f} kWh")
        with col4:
            st.metric("Unmet Demand", f"{allocation_df['Unmet_Demand (kWh)'].sum():.0f} kWh")

        st.dataframe(allocation_df.drop(columns=["Effective_Cost"]), use_container_width=True)

        # Allocation efficiency chart
        st.subheader("📈 Allocation Efficiency")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Demand',
            x=allocation_df['City'],
            y=allocation_df['Total_Demand (kWh)'],
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            name='Received Energy',
            x=allocation_df['City'],
            y=allocation_df['Received_Energy (kWh)'],
            marker_color='green'
        ))
        fig.add_trace(go.Bar(
            name='Unmet Demand',
            x=allocation_df['City'],
            y=allocation_df['Unmet_Demand (kWh)'],
            marker_color='red'
        ))
        fig.update_layout(
            title='Energy Demand vs Allocation by City',
            xaxis_title='City',
            yaxis_title='Energy (kWh)',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Visualize allocation on a map using folium
        st.subheader("Energy Allocation Map")
        m = folium.Map(location=[11.0, 78.0], zoom_start=7)
        for _, row in allocation_df.iterrows():
            city = row["City"]
            lat = geo_data[city]["lat"]
            lon = geo_data[city]["lon"]
            allocated_energy = row["Allocated_Energy (kWh)"]
            tooltip = f"{city}<br>Allocated Energy: {allocated_energy:.2f} kWh"
            folium.Marker(
                location=[lat, lon],
                popup=tooltip,
                tooltip=tooltip,
            ).add_to(m)
        folium_static(m)







# Wind Turbine Hotspot Detection Page
elif page == "Turbine Thermal Defect Detection":
    st.markdown("<h1 class='main-header'>Turbine Thermal Defect Detection</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Upload an image to detect hotspots in wind turbines.
    """)
    
    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        image = np.array(image)  # Convert PIL image to NumPy array
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

        # Allow the user to adjust the target color and tolerance
        st.sidebar.markdown("### Adjust Detection Parameters")
        target_r = 251
        target_g = 243
        target_b = 199
        tolerance = st.sidebar.slider("Tolerance", 0, 100, 30)

        # Define the target RGB color and tolerance range
        target_color = np.array([target_b, target_g, target_r])  # OpenCV uses BGR format
        lower_bound = target_color - tolerance
        upper_bound = target_color + tolerance

        # Clip the lower and upper bounds to ensure valid values
        lower_bound = np.clip(lower_bound, 0, 255)
        upper_bound = np.clip(upper_bound, 0, 255)

        # Create a mask for the selected color range
        mask = cv2.inRange(image, lower_bound, upper_bound)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an overlay image to draw circles
        overlay_img = image.copy()
        min_area = 10  # Minimum area to consider a valid hotspot

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                # Find the minimum enclosing circle for the contour
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                # Draw a green circle on the overlay image
                cv2.circle(overlay_img, center, radius, (0, 255, 0), 2)

        # Convert images back to RGB for display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

        # Display the results
        st.markdown("<h3 class='sub-header'>Results</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        
        with col2:
            st.image(mask, caption="Detected Hotspots (Mask)", use_container_width=True, clamp=True)
        
        with col3:
            st.image(overlay_img, caption="Highlighted Hotspots with Circles", use_container_width=True)


# Renewable Energy Forecaster Page
elif page == "Renewable Energy Forecaster":
    st.markdown("<h1 class='main-header'>Renewable Energy Potential Forecaster</h1>", unsafe_allow_html=True)
    st.markdown("""
    This application forecasts the renewable energy potential for selected districts, 
    helping to identify the best locations for wind, solar, and ocean energy production.
    """)
    
    # Sidebar for inputs
    st.sidebar.header("Settings")
    
    # Input for model directory
    model_dir = "./saved_models"
    
    # Input for forecast days
    forecast_days = st.sidebar.slider(
        "Forecast Days",
        min_value=1,
        max_value=14,
        value=7,
        help="Number of days to forecast (maximum 14 days)"
    )
    
    # Multiselect for districts
    selected_districts = st.sidebar.multiselect(
        "Select Districts",
        options=list(districts.keys()),
        default=["Chennai", "Coimbatore"],
        help="Select one or more districts to forecast"
    )
    
    if not selected_districts:
        st.warning("Please select at least one district to generate forecasts.")
    else:
        # Display district info
        with st.expander("View Selected District Information"):
            district_df = pd.DataFrame.from_dict(
                {k: v for k, v in districts.items() if k in selected_districts}, 
                orient='index'
            )
            district_df['coastal'] = district_df['coastal'].map({True: 'Yes', False: 'No'})
            st.dataframe(district_df)
        
        # Load models
        load_models_button = st.button("Load Models")
        
        if load_models_button or 'models' in st.session_state:
            if 'models' not in st.session_state:
                with st.spinner("Loading prediction models..."):
                    models = load_models(model_dir)
                    if models:
                        st.session_state.models = models
                        st.success("Models loaded successfully!")
                    else:
                        st.error("Failed to load models. Please check the model directory.")
                        
            
            # Generate forecasts button
            forecast_button = st.button("Generate Forecasts")
            
            if forecast_button:
                if len(selected_districts) == 0:
                    st.warning("Please select at least one district to generate forecasts.")
                else:
                    with st.spinner("Generating forecasts..."):
                        district_forecasts = generate_forecasts(
                            selected_districts, 
                            forecast_days, 
                            st.session_state.models
                        )
                        
                        if district_forecasts:
                            st.session_state.district_forecasts = district_forecasts
                            st.session_state.report = generate_forecast_report(district_forecasts)
                            st.success("Forecasts generated successfully!")
                        else:
                            st.error("Failed to generate forecasts. Please try again.")
            
            # Display forecasts if available
            if 'district_forecasts' in st.session_state and 'report' in st.session_state:
                district_forecasts = st.session_state.district_forecasts
                report = st.session_state.report
                
                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["Overview", "District Comparisons", "Detailed Forecasts"])
                
                with tab1:
                    st.header("Overview")
                    
                    # Display forecast period
                    st.subheader("Forecast Period")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Start Date", report['forecast_period']['start'])
                    with col2:
                        st.metric("End Date", report['forecast_period']['end'])
                    
                    # Display energy mix
                    st.subheader("Energy Mix")
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        energy_mix_fig = plot_energy_mix(report)
                        st.plotly_chart(energy_mix_fig, use_container_width=True)
                    
                    with col2:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Wind (%)", f"{report['state_summary']['energy_mix']['wind']:.1f}%")
                        with col2:
                            st.metric("Solar (%)", f"{report['state_summary']['energy_mix']['solar']:.1f}%")
                        with col3:
                            st.metric("Ocean (%)", f"{report['state_summary']['energy_mix']['ocean']:.1f}%")
                    
                    # Display best districts
                    st.subheader("Best Districts")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Best for Wind", report['best_districts']['wind'])
                    with col2:
                        st.metric("Best for Solar", report['best_districts']['solar'])
                    with col3:
                        st.metric("Best for Ocean", report['best_districts']['ocean'])
                
                with tab2:
                    st.header("District Comparisons")
                    
                    # District comparison bar chart
                    comparison_fig = plot_district_comparison(district_forecasts)
                    st.plotly_chart(comparison_fig, use_container_width=True)
                    
                    # Display district rankings
                    st.subheader("District Rankings")
                    
                    ranking_tabs = st.tabs(["Total Energy", "Wind Energy", "Solar Energy", "Ocean Energy"])
                    
                    with ranking_tabs[0]:
                        total_rankings = report['district_rankings']['total_energy']
                        total_df = pd.DataFrame(total_rankings, columns=['District', 'Total Energy (Wh/m²)'])
                        st.dataframe(total_df, use_container_width=True)
                    
                    with ranking_tabs[1]:
                        wind_rankings = report['district_rankings']['wind_energy']
                        wind_df = pd.DataFrame(wind_rankings, columns=['District', 'Wind Energy (Wh/m²)'])
                        st.dataframe(wind_df, use_container_width=True)
                    
                    with ranking_tabs[2]:
                        solar_rankings = report['district_rankings']['solar_energy']
                        solar_df = pd.DataFrame(solar_rankings, columns=['District', 'Solar Energy (Wh/m²)'])
                        st.dataframe(solar_df, use_container_width=True)
                    
                    with ranking_tabs[3]:
                        ocean_rankings = report['district_rankings']['ocean_energy']
                        ocean_df = pd.DataFrame(ocean_rankings, columns=['District', 'Ocean Energy (Wh/m²)'])
                        st.dataframe(ocean_df, use_container_width=True)
                
                with tab3:
                    st.header("Detailed District Forecasts")
                    
                    # District selectbox for detailed view
                    selected_district = st.selectbox(
                        "Select a district for detailed forecast",
                        options=list(district_forecasts.keys())
                    )
                    
                    if selected_district:
                        district_forecast = district_forecasts[selected_district]
                        
                        # Display district metrics
                        st.subheader(f"{selected_district} Energy Potential Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Wind Energy",
                                f"{district_forecast['total_potential']['wind']:.2f} Wh/m²"
                            )
                        
                        with col2:
                            st.metric(
                                "Solar Energy",
                                f"{district_forecast['total_potential']['solar']:.2f} Wh/m²"
                            )
                        
                        with col3:
                            ocean_value = district_forecast['total_potential']['ocean']
                            st.metric(
                                "Ocean Energy",
                                f"{ocean_value:.2f} kWh/m²" if ocean_value > 0 else "N/A"
                            )
                        
                        with col4:
                            st.metric(
                                "Total Energy",
                                f"{district_forecast['total_potential']['total']:.2f} Wh/m²"
                            )
                        
                        # Daily forecast plot
                        st.subheader("Daily Forecast")
                        daily_fig = plot_daily_forecast(district_forecast, selected_district)
                        st.plotly_chart(daily_fig, use_container_width=True)
                        
                        # Hourly pattern plot
                        st.subheader("Hourly Pattern")
                        hourly_fig = plot_hourly_pattern(district_forecast, selected_district)
                        st.plotly_chart(hourly_fig, use_container_width=True)
                        
                        # Hourly data table
                        with st.expander("View Hourly Forecast Data"):
                            hourly_data = district_forecast['hourly_forecast']
                            hourly_data_display = hourly_data.copy()
                            hourly_data_display['datetime'] = hourly_data_display['datetime'].dt.strftime('%Y-%m-%d %H:%M')
                            st.dataframe(hourly_data_display, use_container_width=True)



elif page == "Power Map":
    st.markdown("<h1 class='main-header'>Power Map of Tamil Nadu</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Explore the power map of Tamil Nadu and get insights.
    """)
    
    # Display the soil distribution image
    soil_image_path = r"images_frontend/c.png"  # Replace with the path to your image
    st.image(soil_image_path, caption="Power Map", use_container_width=True)
    
    
    # Add a download button for the soil distribution map (optional)
    with open(soil_image_path, "rb") as file:
        btn = st.download_button(
            label="Download Power Map",
            data=file,
            file_name="power_map.png",
            mime="image/png"
        )

