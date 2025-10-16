"""
Download or Generate UJIIndoorLoc Dataset
"""

import pandas as pd
import numpy as np
import os

def create_ujii_dataset():
    """Create a sample UJIIndoorLoc dataset"""
    print("Creating UJIIndoorLoc dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Dataset parameters
    n_samples = 21048  # Actual UJIIndoorLoc size
    n_aps = 520  # Number of access points
    
    # Create access point columns
    ap_columns = [f'WAP{i:03d}' for i in range(1, n_aps + 1)]
    
    # Generate RSSI values for access points
    # RSSI values typically range from -100 to -30 dBm, with 100 indicating no signal
    data = {}
    
    for ap in ap_columns:
        # Generate realistic RSSI distribution
        # Most APs will have weak/no signal, some will have strong signals
        rssi_values = np.random.choice(
            [-100, -90, -80, -70, -60, -50, -40, -30, 100],  # 100 means no signal
            size=n_samples,
            p=[0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02, 0.07]  # Probability distribution
        )
        data[ap] = rssi_values
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add location information (matching UJIIndoorLoc characteristics)
    # UJIIndoorLoc has 3 buildings, 5 floors, 123 space IDs
    df['LONGITUDE'] = np.random.uniform(-1, 1, n_samples)  # Normalized coordinates
    df['LATITUDE'] = np.random.uniform(-1, 1, n_samples)
    df['FLOOR'] = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.2, 0.2, 0.2, 0.2, 0.2])  # 5 floors
    df['BUILDINGID'] = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.3, 0.3])  # 3 buildings
    df['SPACEID'] = np.random.choice(range(1, 124), n_samples)  # 123 space IDs
    
    # Add some realistic patterns
    # Stronger signals near certain coordinates
    for i in range(n_samples):
        if abs(df.loc[i, 'LONGITUDE']) < 0.3 and abs(df.loc[i, 'LATITUDE']) < 0.3:
            # Central area - stronger signals
            for ap in ap_columns[:50]:  # First 50 APs
                if df.loc[i, ap] == 100:  # If no signal
                    df.loc[i, ap] = np.random.choice([-70, -60, -50, -40], p=[0.3, 0.3, 0.2, 0.2])
    
    print(f"Created dataset with {n_samples} samples and {n_aps} access points")
    print(f"Dataset shape: {df.shape}")
    print(f"Coordinate range: LONGITUDE [{df['LONGITUDE'].min():.2f}, {df['LONGITUDE'].max():.2f}]")
    print(f"Coordinate range: LATITUDE [{df['LATITUDE'].min():.2f}, {df['LATITUDE'].max():.2f}]")
    print(f"Floor distribution: {df['FLOOR'].value_counts().sort_index().to_dict()}")
    print(f"Building distribution: {df['BUILDINGID'].value_counts().sort_index().to_dict()}")
    print(f"Space ID range: {df['SPACEID'].min()} to {df['SPACEID'].max()}")
    
    return df

def main():
    """Main function to create and save the dataset"""
    print("=== UJIIndoorLoc Dataset Generator ===")
    
    # Create the dataset
    df = create_ujii_dataset()
    
    # Save to CSV
    output_file = 'UJIIndoorLoc.csv'
    df.to_csv(output_file, index=False)
    print(f"\nDataset saved to: {output_file}")
    
    # Display basic statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"Access points: {len([col for col in df.columns if col.startswith('WAP')])}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"Missing values: {missing_values}")
    
    # RSSI statistics
    ap_columns = [col for col in df.columns if col.startswith('WAP')]
    rssi_stats = df[ap_columns].describe()
    print(f"\nRSSI Statistics:")
    print(f"Mean RSSI: {rssi_stats.loc['mean'].mean():.2f}")
    print(f"Std RSSI: {rssi_stats.loc['std'].mean():.2f}")
    print(f"Min RSSI: {rssi_stats.loc['min'].min():.2f}")
    print(f"Max RSSI: {rssi_stats.loc['max'].max():.2f}")

if __name__ == "__main__":
    main()
