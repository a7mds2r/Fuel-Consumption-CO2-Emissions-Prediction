import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import set_viz_style, ensure_dir

# Set style for premium visualizations
set_viz_style()

def clean_data() -> None:
    """
    Performs data cleaning on the raw fuel consumption dataset.
    Standardizes columns, removes missing values, duplicates, and outliers.
    Saves the cleaned dataset and diagnostic plots.
    """
    print("--- Starting Task 1: Data Cleaning ---")
    
    # Paths
    raw_path = 'data/raw/Fuel_Consumption_2000-2022.csv'
    processed_dir = 'data/processed'
    figures_dir = 'results/figures'
    
    ensure_dir(processed_dir)
    ensure_dir(figures_dir)
    
    # 1. Load Data
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")
    
    df = pd.read_csv(raw_path)
    raw_shape = df.shape
    print(f"Loaded raw data: {raw_shape[0]} rows, {raw_shape[1]} columns")
    
    # Keep copy for before/after comparison
    df_raw = df.copy()
    
    # 2. Standardize Column Names
    rename_dict = {
        'YEAR': 'YEAR',
        'MAKE': 'MAKE',
        'MODEL': 'MODEL',
        'VEHICLE CLASS': 'VEHICLE_CLASS',
        'ENGINE SIZE': 'ENGINE_SIZE',
        'CYLINDERS': 'CYLINDERS',
        'TRANSMISSION': 'TRANSMISSION',
        'FUEL': 'FUEL',
        'FUEL CONSUMPTION': 'FUEL_CONSUMPTION',
        'HWY (L/100 km)': 'HWY_(L/100_KM)',
        'COMB (L/100 km)': 'COMB_(L/100_KM)',
        'COMB (mpg)': 'COMB_(MPG)',
        'EMISSIONS': 'EMISSIONS'
    }
    df.rename(columns=rename_dict, inplace=True)
    print("Columns standardized to:", list(df.columns))
    
    # 3. Remove Missing Values
    initial_len = len(df)
    df.dropna(inplace=True)
    n_missing = initial_len - len(df)
    print(f"Removed {n_missing} rows with missing values.")
    
    # 4. Remove Duplicates
    initial_len = len(df)
    df.drop_duplicates(inplace=True)
    n_dups = initial_len - len(df)
    print(f"Removed {n_dups} duplicate rows.")
    
    # 5. Remove Invalid Values (Negative or Zero in numeric fields)
    numeric_cols = ['YEAR', 'ENGINE_SIZE', 'CYLINDERS', 'FUEL_CONSUMPTION', 'HWY_(L/100_KM)', 'COMB_(L/100_KM)', 'COMB_(MPG)', 'EMISSIONS']
    initial_len = len(df)
    for col in numeric_cols:
        if col in df.columns:
            df = df[df[col] > 0]
    n_invalid = initial_len - len(df)
    print(f"Removed {n_invalid} rows with negative or zero values in numeric columns.")
    
    # 6. Outlier Removal (IQR for EMISSIONS)
    q1 = df['EMISSIONS'].quantile(0.25)
    q3 = df['EMISSIONS'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    print(f"EMISSIONS IQR Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    initial_len = len(df)
    df_no_outliers = df[(df['EMISSIONS'] >= lower_bound) & (df['EMISSIONS'] <= upper_bound)]
    n_outliers = initial_len - len(df_no_outliers)
    print(f"Removed {n_outliers} outlier rows based on EMISSIONS IQR.")
    df = df_no_outliers
    
    print(f"Final cleaned data size: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 7. Save Cleaned Data
    processed_path = os.path.join(processed_dir, 'Fuel_Consumption_Cleaned.csv')
    df.to_csv(processed_path, index=False)
    print(f"Cleaned data saved to {processed_path}")
    
    # 8. Generate boxplot_cleaning_comparison.png
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('#fafafa')
    
    # Raw emissions
    if 'EMISSIONS' in df_raw.columns:
        sns.boxplot(y=df_raw['EMISSIONS'], ax=axes[0], color='#ff7675', width=0.4)
    else:
        print("Warning: 'EMISSIONS' column not found in raw data for comparison plot.")
    axes[0].set_title('Raw EMISSIONS Distribution (Before Cleaning)', fontsize=12, fontweight='semibold')
    axes[0].set_ylabel('CO2 Emissions (g/km)', fontsize=11)
    axes[0].set_facecolor('#ffffff')
    
    # Cleaned emissions
    sns.boxplot(y=df['EMISSIONS'], ax=axes[1], color='#0984e3', width=0.4)
    axes[1].set_title('Cleaned EMISSIONS Distribution (After Cleaning)', fontsize=12, fontweight='semibold')
    axes[1].set_ylabel('CO2 Emissions (g/km)', fontsize=11)
    axes[1].set_facecolor('#ffffff')
    
    plt.suptitle('Comparison of EMISSIONS Outlier Cleaning (IQR Method)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    boxplot_path = os.path.join(figures_dir, 'boxplot_cleaning_comparison.png')
    plt.savefig(boxplot_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Saved boxplot comparison figure to {boxplot_path}")
    
    # 9. Generate emissions_distribution.png (initial)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#ffffff')
    
    sns.histplot(df['EMISSIONS'], kde=True, bins=30, ax=ax, color='#00b894', edgecolor='white', alpha=0.85)
    ax.set_title('Cleaned CO2 Emissions Distribution Histogram', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('CO2 Emissions (g/km)', fontsize=12)
    ax.set_ylabel('Count / Frequency', fontsize=12)
    
    # Add summary statistics text on plot
    stats_text = (
        f"Mean: {df['EMISSIONS'].mean():.2f}\n"
        f"Median: {df['EMISSIONS'].median():.2f}\n"
        f"Std Dev: {df['EMISSIONS'].std():.2f}\n"
        f"Min: {df['EMISSIONS'].min():.2f}\n"
        f"Max: {df['EMISSIONS'].max():.2f}\n"
        f"N: {len(df)}"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#dfe6e9', alpha=0.9)
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    hist_path = os.path.join(figures_dir, 'emissions_distribution.png')
    plt.savefig(hist_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Saved emissions distribution histogram to {hist_path}")
    
    # Print metrics for summary report
    print("\nSummary metrics for raw vs cleaned:")
    print(f"Raw samples count: {len(df_raw)}")
    print(f"Cleaned samples count: {len(df)}")
    print(f"Percentage remaining: {(len(df) / len(df_raw)) * 100:.2f}%")
    print("--- Task 1 Completed Successfully ---\n")

if __name__ == '__main__':
    clean_data()
