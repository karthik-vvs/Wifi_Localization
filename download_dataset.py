#!/usr/bin/env python3
"""
Dataset Download Script for UJIIndoorLoc
Downloads the real UJIIndoorLoc dataset from UCI Machine Learning Repository.
"""

import pandas as pd
import os
import requests
import zipfile
from io import BytesIO

def download_uji_dataset():
    """Download the real UJIIndoorLoc dataset from UCI ML Repository"""
    print("Downloading real UJIIndoorLoc dataset from UCI ML Repository...")
    
    # UCI ML Repository download URL
    url = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"
    
    try:
        # Download the dataset
        print("Fetching dataset from UCI ML Repository...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Extract the zip file
        print("Extracting dataset...")
        with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
            # List files in the zip
            file_list = zip_file.namelist()
            print(f"Files in archive: {file_list}")
            
            # Extract all files
            zip_file.extractall(".")
            
        print("Dataset downloaded and extracted successfully!")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        print("Falling back to Kaggle download method...")
        return download_from_kaggle()
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return False

def download_from_kaggle():
    """Alternative method to download from Kaggle (requires kaggle API)"""
    try:
        import kaggle
        print("Downloading from Kaggle...")
        kaggle.api.dataset_download_files('giantuji/UjiIndoorLoc', path='.', unzip=True)
        print("Dataset downloaded from Kaggle successfully!")
        return True
    except ImportError:
        print("Kaggle API not installed. Please install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        return False

def combine_uji_files():
    """Combine training and validation data into a single file"""
    print("Combining training and validation data...")
    
    try:
        # Read training and validation data
        train_file = "UJIndoorLoc/trainingData.csv"
        val_file = "UJIndoorLoc/validationData.csv"
        
        if not os.path.exists(train_file):
            print(f"Training file not found: {train_file}")
            return False
            
        if not os.path.exists(val_file):
            print(f"Validation file not found: {val_file}")
            return False
        
        # Load datasets
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Validation data shape: {val_df.shape}")
        
        # Combine datasets
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        print(f"Combined data shape: {combined_df.shape}")
        
        # Save combined dataset
        output_file = 'UJIIndoorLoc.csv'
        try:
            combined_df.to_csv(output_file, index=False)
            print(f"Combined dataset saved to: {output_file}")
        except PermissionError:
            print(f"Permission denied writing to {output_file}. Trying alternative filename...")
            output_file = 'UJIIndoorLoc_combined.csv'
            combined_df.to_csv(output_file, index=False)
            print(f"Combined dataset saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error combining datasets: {e}")
        return False

def normalize_coordinates(df):
    """Normalize coordinates to [-1, 1] range for better training"""
    print("Normalizing coordinates...")
    
    # Original coordinate ranges from UCI description:
    # Longitude: -7695.9387549299299000 to -7299.786516730871000
    # Latitude: 4864745.7450159714 to 4865017.3646842018
    
    # Normalize longitude to [-1, 1]
    lon_min, lon_max = df['LONGITUDE'].min(), df['LONGITUDE'].max()
    df['LONGITUDE'] = 2 * (df['LONGITUDE'] - lon_min) / (lon_max - lon_min) - 1
    
    # Normalize latitude to [-1, 1]
    lat_min, lat_max = df['LATITUDE'].min(), df['LATITUDE'].max()
    df['LATITUDE'] = 2 * (df['LATITUDE'] - lat_min) / (lat_max - lat_min) - 1
    
    print(f"Longitude range: {df['LONGITUDE'].min():.3f} to {df['LONGITUDE'].max():.3f}")
    print(f"Latitude range: {df['LATITUDE'].min():.3f} to {df['LATITUDE'].max():.3f}")
    
    return df

def main():
    """Main function to download and prepare the dataset"""
    print("=== UJIIndoorLoc Dataset Downloader ===")
    
    # Check if dataset already exists
    if os.path.exists('UJIIndoorLoc.csv'):
        print("UJIIndoorLoc.csv already exists!")
        df = pd.read_csv('UJIIndoorLoc.csv')
        print(f"Dataset shape: {df.shape}")
        return
    
    # Download dataset
    if download_uji_dataset():
        # Combine training and validation data
        if combine_uji_files():
            # Load and normalize coordinates
            df = pd.read_csv('UJIIndoorLoc.csv')
            df = normalize_coordinates(df)
            try:
                df.to_csv('UJIIndoorLoc.csv', index=False)
            except PermissionError:
                print("Permission denied. Dataset is ready but couldn't overwrite the file.")
                print("The normalized dataset is available in memory.")
            
            print("\n=== Dataset Statistics ===")
            print(f"Total samples: {len(df)}")
            print(f"WiFi Access Points: {len([col for col in df.columns if col.startswith('WAP')])}")
            print(f"Longitude range: {df['LONGITUDE'].min():.3f} to {df['LONGITUDE'].max():.3f}")
            print(f"Latitude range: {df['LATITUDE'].min():.3f} to {df['LATITUDE'].max():.3f}")
            print(f"Floor distribution: {df['FLOOR'].value_counts().sort_index().to_dict()}")
            print(f"Building distribution: {df['BUILDINGID'].value_counts().sort_index().to_dict()}")
            print(f"Space ID range: {df['SPACEID'].min()} to {df['SPACEID'].max()}")
            
            print(f"\nReal UJIIndoorLoc dataset ready for WiFi Indoor Localization experiments!")
        else:
            print("Failed to combine dataset files.")
    else:
        print("Failed to download dataset. Please check your internet connection.")
        print("You can manually download from:")
        print("1. UCI ML Repository: https://archive.ics.uci.edu/dataset/310/ujiindoorloc")
        print("2. Kaggle: https://www.kaggle.com/datasets/giantuji/UjiIndoorLoc")

if __name__ == "__main__":
    main()