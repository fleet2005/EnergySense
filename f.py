import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
import requests
from datetime import datetime, timedelta
import sys
import time

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
            st.error("Failed to load required models. Please check if model files exist in the specified directory.")
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
        yaxis_title='Energy Potential (kWh/m²)',
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
        yaxis_title='Energy Potential (kWh/m²)',
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
        yaxis_title='Average Daily Energy Potential (kWh/m²)',
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

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Renewable Energy Forecaster",
        page_icon="🌞",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("🔮 Renewable Energy Potential Forecaster")
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
        max_value=16,
        value=7,
        help="Number of days to forecast (maximum 16 days)"
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
        return
    
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
                    return
        
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
                    total_df = pd.DataFrame(total_rankings, columns=['District', 'Total Energy (kWh/m²)'])
                    st.dataframe(total_df, use_container_width=True)
                
                with ranking_tabs[1]:
                    wind_rankings = report['district_rankings']['wind_energy']
                    wind_df = pd.DataFrame(wind_rankings, columns=['District', 'Wind Energy (kWh/m²)'])
                    st.dataframe(wind_df, use_container_width=True)
                
                with ranking_tabs[2]:
                    solar_rankings = report['district_rankings']['solar_energy']
                    solar_df = pd.DataFrame(solar_rankings, columns=['District', 'Solar Energy (kWh/m²)'])
                    st.dataframe(solar_df, use_container_width=True)
                
                with ranking_tabs[3]:
                    ocean_rankings = report['district_rankings']['ocean_energy']
                    ocean_df = pd.DataFrame(ocean_rankings, columns=['District', 'Ocean Energy (kWh/m²)'])
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
                            f"{district_forecast['total_potential']['wind']:.2f} kWh/m²"
                        )
                    
                    with col2:
                        st.metric(
                            "Solar Energy",
                            f"{district_forecast['total_potential']['solar']:.2f} kWh/m²"
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
                            f"{district_forecast['total_potential']['total']:.2f} kWh/m²"
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

if __name__ == "__main__":
    main()