#!/usr/bin/env python3
"""
Script to retrain the renewable energy models with the current scikit-learn version.
Run this script if you encounter model loading errors due to version compatibility issues.
"""

import os
import sys
from model import main

def retrain_models():
    """Retrain all renewable energy models"""
    print("🔄 Retraining renewable energy models...")
    print("This will create new model files compatible with your current scikit-learn version.")
    print()
    
    # Check if saved_models directory exists
    if not os.path.exists('saved_models'):
        print("❌ saved_models directory not found. Creating it...")
        os.makedirs('saved_models')
    
    try:
        # Run the main training function from model.py
        main()
        print()
        print("✅ Models retrained successfully!")
        print("You can now use the Renewable Energy Forecaster without issues.")
        
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        print("Please check your internet connection and try again.")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("EnergySense - Model Retraining Script")
    print("=" * 60)
    print()
    
    success = retrain_models()
    
    if success:
        print()
        print("🎉 All done! Your models are now ready to use.")
    else:
        print()
        print("💥 Model retraining failed. Please check the error messages above.")
        sys.exit(1)
