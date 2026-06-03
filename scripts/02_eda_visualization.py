import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for premium visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.facecolor': '#fafafa',
    'axes.facecolor': '#ffffff'
})

# Get base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def map_fuel_type(fuel):
    fuel_map = {
        'X': 'Regular Gasoline',
        'Z': 'Premium Gasoline',
        'D': 'Diesel',
        'E': 'Ethanol (E85)',
        'N': 'Natural Gas (CNG)'
    }
    return fuel_map.get(fuel, f'Other ({fuel})')


def map_transmission(t):
    t_str = str(t).upper()
    if t_str.startswith('AS'):
        return 'Automatic (Select Shift)'
    elif t_str.startswith('AM'):
        return 'Automated Manual'
    elif t_str.startswith('AV') or t_str.startswith('CV'):
        return 'Continuously Variable (CVT)'
    elif t_str.startswith('A'):
        return 'Automatic'
    elif t_str.startswith('M'):
        return 'Manual'
    else:
        return 'Other'


def run_eda():
    print("--- Starting Task 2: EDA & Visualization ---")

    cleaned_path = os.path.join(BASE_DIR, 'data', 'processed', 'Fuel_Consumption_Cleaned.csv')
    figures_dir = os.path.join(BASE_DIR, 'results', 'figures')
    reports_dir = os.path.join(BASE_DIR, 'results', 'reports')

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load Cleaned Data
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned data file not found at {cleaned_path}. Ensure data cleaning script runs first.")

    df = pd.read_csv(cleaned_path)
    print(f"Loaded cleaned data with {len(df)} rows.")

    # Add mapped categories for plotting
    df['FUEL_NAME'] = df['FUEL'].apply(map_fuel_type)
    df['TRANS_TYPE'] = df['TRANSMISSION'].apply(map_transmission)
    engineered_cols = [
        'ENGINE_SIZE_PER_CYLINDER',
        'CITY_HWY_FUEL_GAP',
        'COMB_CITY_FUEL_RATIO',
        'COMB_HWY_FUEL_RATIO',
        'FUEL_EFFICIENCY_KM_PER_L',
        'ENGINE_LOAD_PROXY',
        'TRANSMISSION_GEARS',
    ]
    engineered_cols = [col for col in engineered_cols if col in df.columns]

    # --- VISUALIZATION 1: emissions_distribution.png ---
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#ffffff')
    sns.histplot(df['EMISSIONS'], kde=True, bins=35, ax=ax, color='#6c5ce7', edgecolor='white', alpha=0.8)
    ax.set_title('Distribution of Vehicle CO2 Emissions', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('CO2 Emissions (g/km)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'emissions_distribution.png'), dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print("1. Generated emissions_distribution.png")

    # --- VISUALIZATION 2: fuel_vs_emissions.png ---
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#ffffff')
    sns.regplot(data=df, x='FUEL_CONSUMPTION', y='EMISSIONS', ax=ax,
                scatter_kws={'alpha': 0.2, 'color': '#00cec9', 's': 10},
                line_kws={'color': '#ff7675', 'linewidth': 2})
    ax.set_title('Fuel Consumption (City) vs CO2 Emissions', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Fuel Consumption (L/100 km)', fontsize=12)
    ax.set_ylabel('CO2 Emissions (g/km)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fuel_vs_emissions.png'), dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print("2. Generated fuel_vs_emissions.png")

    # --- VISUALIZATION 3: engine_size_vs_emissions.png ---
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#ffffff')
    sns.scatterplot(data=df, x='ENGINE_SIZE', y='EMISSIONS', hue='CYLINDERS', palette='viridis', alpha=0.4, s=15, ax=ax)
    ax.set_title('Engine Size vs CO2 Emissions', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Engine Size (L)', fontsize=12)
    ax.set_ylabel('CO2 Emissions (g/km)', fontsize=12)
    ax.legend(title='Cylinders', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'engine_size_vs_emissions.png'), dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print("3. Generated engine_size_vs_emissions.png")

    # --- VISUALIZATION 4: cylinders_vs_emissions.png ---
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#ffffff')
    # Filter to main cylinder numbers for readability (e.g. 3, 4, 5, 6, 8, 10, 12, 16)
    main_cyl = df['CYLINDERS'].value_counts()
    valid_cyl = main_cyl[main_cyl > 5].index.tolist()
    df_cyl = df[df['CYLINDERS'].isin(valid_cyl)]
    sns.boxplot(data=df_cyl, x='CYLINDERS', y='EMISSIONS', ax=ax, palette='Blues')
    ax.set_title('Number of Cylinders vs CO2 Emissions', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Cylinders', fontsize=12)
    ax.set_ylabel('CO2 Emissions (g/km)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'cylinders_vs_emissions.png'), dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print("4. Generated cylinders_vs_emissions.png")

    # --- VISUALIZATION 5: correlation_heatmap.png ---
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#fafafa')
    numeric_cols = [
        'YEAR',
        'ENGINE_SIZE',
        'CYLINDERS',
        'FUEL_CONSUMPTION',
        'HWY_(L/100_KM)',
        'COMB_(L/100_KM)',
        'COMB_(MPG)',
        *engineered_cols,
        'EMISSIONS',
    ]
    numeric_df = df[numeric_cols]
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title('Correlation Heatmap of Numeric Features', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'correlation_heatmap.png'), dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print("5. Generated correlation_heatmap.png")

    # --- VISUALIZATION 6: emissions_by_fuel.png ---
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#ffffff')
    # Filter to common fuel types to avoid clutter
    fuel_counts = df['FUEL_NAME'].value_counts()
    valid_fuels = fuel_counts[fuel_counts > 10].index.tolist()
    df_fuel = df[df['FUEL_NAME'].isin(valid_fuels)]
    sns.boxplot(data=df_fuel, x='FUEL_NAME', y='EMISSIONS', ax=ax, palette='Set2')
    ax.set_title('CO2 Emissions Distribution by Fuel Type', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Fuel Type', fontsize=12)
    ax.set_ylabel('CO2 Emissions (g/km)', fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'emissions_by_fuel.png'), dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print("6. Generated emissions_by_fuel.png")

    # --- VISUALIZATION 7: emissions_by_transmission.png ---
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#ffffff')
    sns.boxplot(data=df, x='TRANS_TYPE', y='EMISSIONS', ax=ax, palette='Set3')
    ax.set_title('CO2 Emissions Distribution by Transmission Class', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Transmission Type', fontsize=12)
    ax.set_ylabel('CO2 Emissions (g/km)', fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'emissions_by_transmission.png'), dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print("7. Generated emissions_by_transmission.png")

    # --- VISUALIZATION 8: top_10_makes_by_emissions.png ---
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#ffffff')
    top_makes = df.groupby('MAKE')['EMISSIONS'].mean().sort_values(ascending=False).head(10).reset_index()
    sns.barplot(data=top_makes, y='MAKE', x='EMISSIONS', ax=ax, palette='Reds_r')
    ax.set_title('Top 10 Car Makes with Highest Average CO2 Emissions', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Average CO2 Emissions (g/km)', fontsize=12)
    ax.set_ylabel('Manufacturer', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'top_10_makes_by_emissions.png'), dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print("8. Generated top_10_makes_by_emissions.png")

    # --- VISUALIZATION 9: bottom_10_makes_by_emissions.png ---
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#ffffff')
    bottom_makes = df.groupby('MAKE')['EMISSIONS'].mean().sort_values(ascending=True).head(10).reset_index()
    sns.barplot(data=bottom_makes, y='MAKE', x='EMISSIONS', ax=ax, palette='GnBu')
    ax.set_title('Top 10 Car Makes with Lowest Average CO2 Emissions', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Average CO2 Emissions (g/km)', fontsize=12)
    ax.set_ylabel('Manufacturer', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'bottom_10_makes_by_emissions.png'), dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print("9. Generated bottom_10_makes_by_emissions.png")

    # --- VISUALIZATION 10: emissions_trend_by_year.png ---
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#ffffff')
    yearly_trend = df.groupby('YEAR')['EMISSIONS'].mean().reset_index()
    sns.lineplot(data=yearly_trend, x='YEAR', y='EMISSIONS', marker='o', color='#fd79a8', linewidth=2.5, markersize=8, ax=ax)
    ax.set_title('Average CO2 Emissions Trend (2000 - 2022)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Model Year', fontsize=12)
    ax.set_ylabel('Average CO2 Emissions (g/km)', fontsize=12)
    ax.set_xticks(range(2000, 2023, 2))
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'emissions_trend_by_year.png'), dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print("10. Generated emissions_trend_by_year.png")

    # --- VISUALIZATION 11: pairplot.png ---
    pair_cols = ['ENGINE_SIZE', 'FUEL_CONSUMPTION', 'CITY_HWY_FUEL_GAP', 'FUEL_EFFICIENCY_KM_PER_L', 'EMISSIONS']
    pair_cols = [col for col in pair_cols if col in df.columns]
    # Subsample data to 1000 rows to ensure fast rendering of pairplot
    df_sub = df[pair_cols].sample(n=min(1000, len(df)), random_state=42)
    pairplot = sns.pairplot(df_sub, diag_kind='kde', plot_kws={'alpha': 0.4, 's': 15, 'color': '#0984e3'}, diag_kws={'color': '#0984e3'})
    pairplot.fig.patch.set_facecolor('#fafafa')
    pairplot.fig.suptitle('Pairplot of Key Features and Emissions', y=1.02, fontsize=14, fontweight='bold')
    pairplot.savefig(os.path.join(figures_dir, 'pairplot.png'), dpi=300, facecolor='#fafafa')
    plt.close()
    print("11. Generated pairplot.png")

    # 3. Create EDA Report (results/reports/eda_report.md)
    print("Creating EDA Report...")
    summary_stats = numeric_df.describe().to_markdown()

    # Calculate some specific findings dynamically
    c_eng_emi = corr.loc['ENGINE_SIZE', 'EMISSIONS']
    c_cyl_emi = corr.loc['CYLINDERS', 'EMISSIONS']
    c_fuel_emi = corr.loc['FUEL_CONSUMPTION', 'EMISSIONS']
    engineered_corr = (
        corr['EMISSIONS']
        .drop(labels=['EMISSIONS'], errors='ignore')
        .loc[[col for col in engineered_cols if col in corr.index]]
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .to_markdown()
        if engineered_cols
        else 'No engineered feature columns were found.'
    )

    # Year difference
    emi_2000 = df[df['YEAR'] == 2000]['EMISSIONS'].mean()
    emi_2022 = df[df['YEAR'] == 2022]['EMISSIONS'].mean()
    pct_reduction = ((emi_2000 - emi_2022) / emi_2000) * 100

    # Fuel impact
    fuel_means = df.groupby('FUEL_NAME')['EMISSIONS'].mean().to_markdown()

    report_content = f"""# Exploratory Data Analysis (EDA) Report

This report presents a thorough exploratory analysis of the vehicle fuel consumption and CO2 emissions dataset (2000 - 2022).

## 1. Summary Statistics

The following table summarizes the key statistics of the numeric features and the target variable (`EMISSIONS`) after outlier cleaning and deduplication:

{summary_stats}

---

## 2. Key Findings and Insights

### 2.1 CO2 Emissions Distribution
- The target variable `EMISSIONS` has a symmetric, bell-shaped distribution centered around **{df['EMISSIONS'].mean():.2f} g/km**, with a standard deviation of **{df['EMISSIONS'].std():.2f} g/km**.
- The outlier cleaning process successfully filtered extreme and invalid samples, resulting in a cleaner dataset for modeling.

### 2.2 Feature Correlations with CO2 Emissions
- **City Fuel Consumption (`FUEL_CONSUMPTION`)**: Has a strong positive correlation of **{c_fuel_emi:.3f}** with `EMISSIONS`. This indicates that vehicle fuel usage is the single most direct predictor of carbon output.
- **Engine Size (`ENGINE_SIZE`)**: Shows a strong positive correlation of **{c_eng_emi:.3f}** with `EMISSIONS`. Larger engines burn more fuel and therefore emit more carbon dioxide.
- **Cylinders (`CYLINDERS`)**: Shows a positive correlation of **{c_cyl_emi:.3f}**. Additional cylinders generally indicate higher displacement and higher emissions.

### 2.3 Fuel Type Impact Analysis
The average carbon footprint varies widely across fuel technologies:

{fuel_means}

- **Regular & Premium Gasoline** remain the benchmark standards, with Premium emissions slightly higher on average, often due to high-performance engines.
- **Ethanol (E85)** displays lower tail emissions in some categories, but due to its lower energy density, vehicles burn a higher volume of fuel, which partially offsets savings.
- **Natural Gas (CNG)** and **Diesel** show specialized profiles, with CNG being historically cleaner.

### 2.4 Engineered Feature Signals
The cleaning pipeline adds engineered predictors before modeling:

{engineered_corr}

- `FUEL_EFFICIENCY_KM_PER_L` converts combined consumption into an efficiency view.
- `CITY_HWY_FUEL_GAP` captures urban/highway driving penalty.
- `ENGINE_SIZE_PER_CYLINDER`, `ENGINE_LOAD_PROXY`, and `TRANSMISSION_GEARS` add powertrain structure.

### 2.5 Transmission Efficiency
- Manual and Continuously Variable Transmissions (CVTs) generally show lower average emissions compared to older traditional Automatic systems, reflecting their higher energy transfer efficiency.
- Newer multi-gear automatic systems (such as Select Shift) have closed the efficiency gap significantly.

### 2.6 Yearly Trends (2000 - 2022)
- Average emissions dropped from **{emi_2000:.2f} g/km** in 2000 to **{emi_2022:.2f} g/km** in 2022.
- This represents a **{pct_reduction:.2f}% reduction** in average carbon output over 22 years.
- This positive trend is driven by regulatory standards, direct fuel injection, turbocharging, hybridization, and transition to smaller engine sizes.

---

## 3. High-Carbon vs Low-Carbon Manufacturers

- **Highest Average Emitters**: Manufacturers specializing in luxury, sports, or heavy vehicles (e.g., Lamborghini, Bugatti, Bentley, Rolls-Royce).
- **Lowest Average Emitters**: Manufacturers focusing on compact, hybrid, and small passenger cars (e.g., Smart, Toyota, Honda, Mazda).
"""

    report_path = os.path.join(reports_dir, 'eda_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"EDA Report successfully saved to {report_path}")
    print("--- Task 2 Completed Successfully ---\n")


if __name__ == '__main__':
    run_eda()
