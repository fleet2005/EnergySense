#!/bin/bash

# EnergySense deployment setup script
echo "🚀 Setting up EnergySense..."

# Create saved_models directory if it doesn't exist
mkdir -p saved_models

# Check if models exist, if not retrain them
if [ ! -f "saved_models/wind_model.pkl" ] || [ ! -f "saved_models/solar_model.pkl" ]; then
    echo "📊 Training ML models..."
    python retrain_models.py
else
    echo "✅ Models already exist, skipping training..."
fi

echo "🎉 EnergySense setup complete!"
