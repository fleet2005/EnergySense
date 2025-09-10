# 🌞 EnergySense - Renewable Energy Management System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42.2-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

EnergySense is a comprehensive renewable energy management platform that provides intelligent forecasting, optimization, and monitoring capabilities for renewable energy systems across Tamil Nadu, India.

## 🚀 Features

### 📊 Renewable Energy Forecaster
- **Multi-Energy Prediction**: Forecasts wind, solar, and ocean energy potential
- **Machine Learning Models**: Uses RandomForest and GradientBoosting algorithms
- **District Coverage**: Supports 8 major Tamil Nadu districts
- **Interactive Visualizations**: Real-time charts and energy mix analysis
- **Weather Integration**: Open-Meteo API for accurate weather data

### 🗺️ Grid Energy Allocation
- **Smart Optimization**: Linear programming-based energy distribution
- **Multi-City Support**: Optimizes energy allocation across 8 Tamil Nadu cities
- **Constraint Handling**: Considers transmission losses, capacity limits, and demand
- **Interactive Maps**: Folium-based visualization of energy allocation
- **Cost Minimization**: Optimizes for transmission losses and distribution costs

### 🔍 Turbine Thermal Defect Detection
- **Computer Vision**: OpenCV-based hotspot detection in wind turbines
- **Real-time Analysis**: Upload and analyze thermal images instantly
- **Adjustable Parameters**: Customizable detection sensitivity
- **Visual Results**: Original image, mask, and highlighted hotspots

### 🗺️ Power Map
- **Geographic Visualization**: Tamil Nadu power distribution map
- **Download Feature**: Export power maps for offline use

## 🏗️ Architecture

```
EnergySense/
├── app.py                 # Main Streamlit application
├── model.py              # ML model training and data processing
├── prediction.py         # Forecasting engine and model loading
├── opt.py               # Energy allocation optimization
├── retrain_models.py    # Model retraining utility
├── requirements.txt     # Python dependencies
├── runtime.txt         # Python version specification
├── saved_models/       # Pre-trained ML models
│   ├── wind_model.pkl
│   ├── solar_model.pkl
│   ├── ocean_model.pkl
│   └── scalers/
├── images_frontend/    # UI assets
└── test_images/       # Sample images for testing
```

## 🛠️ Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Local Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd EnergySense
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Retrain models** (if needed)
   ```bash
   python retrain_models.py
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

### 🚀 Deployment (Heroku)

1. **Create a Heroku app**
   ```bash
   heroku create your-energysense-app
   ```

2. **Use minimal requirements for better compatibility**
   ```bash
   cp requirements-minimal.txt requirements.txt
   ```

3. **Deploy to Heroku**
   ```bash
   git add .
   git commit -m "Deploy EnergySense"
   git push heroku main
   ```

4. **Open the app**
   ```bash
   heroku open
   ```

**Note**: If you encounter Python 3.13 compatibility issues, the runtime.txt is set to Python 3.12.7 for better stability.

### 🌐 Alternative Deployment (Streamlit Cloud)

1. **Fork this repository**
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub account**
4. **Select your forked repository**
5. **Deploy!**

## 🎯 Usage

### Starting the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Navigation
- **Home**: Overview of all features
- **Renewable Energy Forecaster**: Generate energy predictions
- **Grid Energy Allocation**: Optimize energy distribution
- **Turbine Thermal Defect Detection**: Analyze thermal images
- **Power Map**: View Tamil Nadu power distribution

### Renewable Energy Forecasting
1. Select districts from the sidebar
2. Choose forecast period (1-14 days)
3. Click "Load Models" to initialize ML models
4. Click "Generate Forecasts" to create predictions
5. View results in interactive charts and tables

### Grid Energy Allocation
1. Set grid parameters in the sidebar
2. Click "Allocate Energy to Cities"
3. View optimization results and interactive map

### Thermal Defect Detection
1. Upload a thermal image (JPG/PNG)
2. Adjust detection parameters if needed
3. View detection results with highlighted hotspots

## 🔧 Configuration

### Model Retraining
If you encounter model loading errors due to scikit-learn version compatibility:

```bash
python retrain_models.py
```

This will retrain all models with your current scikit-learn version.

### Custom Districts
To add new districts, modify the `districts` dictionary in `app.py`:

```python
districts = {
    'YourDistrict': {'lat': 12.3456, 'lon': 78.9012, 'coastal': False},
    # ... existing districts
}
```

## 📊 Supported Districts

| District | Latitude | Longitude | Coastal |
|----------|----------|-----------|---------|
| Chennai | 13.0827 | 80.2707 | Yes |
| Coimbatore | 11.0168 | 76.9558 | No |
| Madurai | 9.9252 | 78.1198 | No |
| Trichy | 10.7905 | 78.7047 | No |
| Salem | 11.6643 | 78.1460 | No |
| Vellore | 12.9165 | 79.1325 | No |
| Kanyakumari | 8.0883 | 77.5385 | Yes |
| Tuticorin | 8.7642 | 78.1348 | Yes |

## 🧠 Machine Learning Models

### Wind Energy Model
- **Algorithm**: RandomForest Regressor
- **Features**: Wind speed (10m, 80m, 120m), gusts, air density, pressure
- **Performance**: R² = 0.97, MSE = 3702.49

### Solar Energy Model
- **Algorithm**: GradientBoosting Regressor
- **Features**: Temperature, cloud cover, solar zenith angle, irradiance
- **Performance**: R² = 0.99, MSE = 50.68

### Ocean Energy Model
- **Algorithm**: RandomForest Regressor
- **Features**: Wave height, tide factors, wind conditions
- **Performance**: R² = 0.91, MSE = 49310.57
- **Note**: Only available for coastal districts

## 🔌 API Integrations

- **Open-Meteo API**: Weather data and forecasts
- **Folium**: Interactive map visualizations
- **Streamlit**: Web application framework

## 📈 Performance Metrics

- **Model Accuracy**: 91-99% R² scores across all energy types
- **Response Time**: < 2 seconds for forecast generation
- **Scalability**: Supports up to 8 districts simultaneously
- **Data Coverage**: 3 months of historical weather data

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   python retrain_models.py
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Port Already in Use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

4. **API Connection Issues**
   - Check internet connection
   - Models will use synthetic data as fallback

5. **Python 3.13 Compatibility Issues**
   ```bash
   # Use Python 3.12 instead
   echo "python-3.12.7" > runtime.txt
   cp requirements-minimal.txt requirements.txt
   ```

6. **Deployment Build Failures**
   ```bash
   # Try with minimal requirements
   cp requirements-minimal.txt requirements.txt
   # Or use specific versions
   pip install --upgrade pip setuptools wheel
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Open-Meteo** for weather data API
- **Streamlit** for the web framework
- **scikit-learn** for machine learning algorithms
- **Folium** for interactive maps
- **Tamil Nadu Government** for energy resource data

## 📞 Support

For support, email vishalsrinivasancontact@gmail.com or create an issue in the repository.

## 🔮 Future Enhancements

- [ ] Real-time IoT sensor integration
- [ ] Mobile application
- [ ] Advanced machine learning models
- [ ] Multi-state support
- [ ] Energy trading platform
- [ ] Carbon footprint tracking

---

**Made with ❤️ by Vishal Srinivasan for sustainable energy management**
