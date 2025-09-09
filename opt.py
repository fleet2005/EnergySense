import streamlit as st
import pandas as pd
import pulp
import folium
from streamlit_folium import folium_static
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Advanced Energy Allocation", layout="wide")

# Title
st.title("Advanced Energy Allocation to Cities")

# List of cities in Tamil Nadu
cities = ["Chennai", "Coimbatore", "Madurai", "Salem", "Thoothukudi", "Dindigul", "Nagapattinam", "Ramanathapuram"]

# Updated city demand data (in kWh)
city_demand = {
    "Chennai": {"residential": 1100, "commercial": 825, "industrial": 825},
    "Coimbatore": {"residential": 600, "commercial": 450, "industrial": 450},
    "Madurai": {"residential": 400, "commercial": 300, "industrial": 300},
    "Salem": {"residential": 340, "commercial": 255, "industrial": 255},
    "Thoothukudi": {"residential": 300, "commercial": 225, "industrial": 225},
    "Dindigul": {"residential": 280, "commercial": 210, "industrial": 210},
    "Nagapattinam": {"residential": 200, "commercial": 150, "industrial": 150},
    "Ramanathapuram": {"residential": 180, "commercial": 135, "industrial": 135},
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

# User input for grid parameters
st.sidebar.header("Grid Parameters Input")

# Input fields for grid parameters
total_solar_gen = st.sidebar.number_input("Total Solar Generation (kWh)", min_value=0, value=5000)
total_wind_gen = st.sidebar.number_input("Total Wind Generation (kWh)", min_value=0, value=3000)
total_battery_level = st.sidebar.number_input("Total Battery Level (%)", min_value=0, max_value=100, value=80)
energy_storage_capacity = st.sidebar.number_input("Energy Storage Capacity (kWh)", min_value=0, value=2000)
grid_frequency = st.sidebar.number_input("Grid Frequency (Hz)", min_value=49.0, max_value=51.0, value=50.0)

# Calculate total energy available
total_energy_available = total_solar_gen + total_wind_gen + (total_battery_level * 0.1)  # Assume 10% of battery is used

# Display total energy available
st.sidebar.write(f"Total Energy Available: {total_energy_available:.2f} kWh")

# Button to run allocation
if st.sidebar.button("Allocate Energy to Cities"):
    # Define the optimization problem
    prob = pulp.LpProblem("Energy_Allocation", pulp.LpMinimize)

    # Decision variables: Energy allocated to each city
    allocation = pulp.LpVariable.dicts("Allocation", cities, lowBound=0)

    # Slack variables for unmet demand
    unmet_demand = pulp.LpVariable.dicts("Unmet_Demand", cities, lowBound=0)

    # Battery charge/discharge variables
    battery_charge = pulp.LpVariable("Battery_Charge", lowBound=0, upBound=energy_storage_capacity)
    battery_discharge = pulp.LpVariable("Battery_Discharge", lowBound=0, upBound=energy_storage_capacity)

    # Objective function: Minimize total cost (transmission losses + distribution costs + grid instability penalties)
    prob += (
        pulp.lpSum(allocation[city] * transmission_losses[city] for city in cities)  # Transmission losses
        + pulp.lpSum(allocation[city] * distribution_costs[city] for city in cities)  # Distribution costs
        + pulp.lpSum(unmet_demand[city] * 1000 for city in cities)  # Penalty for unmet demand (high cost)
    ), "Total_Cost"

    # Constraints
    # 1. Total energy allocated cannot exceed total energy available
    prob += pulp.lpSum(allocation[city] for city in cities) + battery_charge <= total_energy_available, "Total_Energy_Constraint"

    # 2. Meet demand for each city (considering transmission losses and unmet demand)
    for city in cities:
        total_demand = (
            city_demand[city]["residential"]
            + city_demand[city]["commercial"]
            + city_demand[city]["industrial"]
        )
        prob += (
            allocation[city] * (1 - transmission_losses[city]) + unmet_demand[city] >= total_demand
        ), f"Demand_Constraint_{city}"

    # 3. Transmission line capacity constraints
    for city in cities:
        prob += allocation[city] <= transmission_capacity[city], f"Transmission_Capacity_{city}"

    # 4. Battery constraints
    prob += battery_charge <= energy_storage_capacity, "Battery_Charge_Limit"
    prob += battery_discharge <= energy_storage_capacity, "Battery_Discharge_Limit"

    # 5. Grid frequency stability constraints
    total_demand = sum(
        city_demand[city]["residential"] + city_demand[city]["commercial"] + city_demand[city]["industrial"]
        for city in cities
    )
    total_supply = total_solar_gen + total_wind_gen + battery_discharge - battery_charge
    frequency_deviation = (total_supply - total_demand) / total_demand  # Proportional deviation
    prob += frequency_deviation >= -0.01, "Frequency_Lower_Limit"  # Allow 1% drop
    prob += frequency_deviation <= 0.01, "Frequency_Upper_Limit"  # Allow 1% rise

    # Solve the problem
    prob.solve()

    # Check if the problem was solved successfully
    if pulp.LpStatus[prob.status] == "Optimal":
        # Extract results
        allocation_results = []
        for city in cities:
            allocated_energy = allocation[city].varValue
            total_demand = (
                city_demand[city]["residential"]
                + city_demand[city]["commercial"]
                + city_demand[city]["industrial"]
            )
            unmet = unmet_demand[city].varValue
            allocation_results.append({
                "City": city,
                "Residential_Demand (kWh)": city_demand[city]["residential"],
                "Commercial_Demand (kWh)": city_demand[city]["commercial"],
                "Industrial_Demand (kWh)": city_demand[city]["industrial"],
                "Total_Demand (kWh)": total_demand,
                "Allocated_Energy (kWh)": allocated_energy,
                "Transmission_Loss (kWh)": allocated_energy * transmission_losses[city],
                "Received_Energy (kWh)": allocated_energy * (1 - transmission_losses[city]),
                "Unmet_Demand (kWh)": unmet,
            })

        # Create a DataFrame for results
        allocation_df = pd.DataFrame(allocation_results)

        # Display results
        st.subheader("Energy Allocation Results")
        st.dataframe(allocation_df)

        # Visualize allocation on a map using folium
        st.subheader("Energy Allocation Map")

        # Create a folium map centered on Tamil Nadu
        m = folium.Map(location=[11.0, 78.0], zoom_start=7)

        # Add markers for each city
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

        # Display the map
        folium_static(m)
    else:
        st.error("Optimization failed! Check input parameters and constraints.")



# Wind Turbine Hotspot Detection Page
elif page == "Wind Turbine Hotspot Detection":
    st.markdown("<h1 class='main-header'>Wind Turbine Hotspot Detection</h1>", unsafe_allow_html=True)
    
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

        # Define the target RGB color (251, 243, 199) and tolerance range
        target_color = np.array([251, 243, 199])
        tolerance = 30  # Adjust the tolerance to control how close the colors should be

        # Create a mask for pixels close to the target color
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
            st.image(image, caption="Original Image", use_column_width=True)
        
        with col2:
            st.image(mask, caption="Detected Hotspots (Mask)", use_column_width=True, clamp=True)
        
        with col3:
            st.image(overlay_img, caption="Highlighted Hotspots with Circles", use_column_width=True)