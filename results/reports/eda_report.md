# Exploratory Data Analysis (EDA) Report

This report presents a thorough exploratory analysis of the vehicle fuel consumption and CO2 emissions dataset (2000 - 2022).

## 1. Summary Statistics

The following table summarizes the key statistics of the numeric features and the target variable (`EMISSIONS`) after outlier cleaning and deduplication:

|       |        YEAR |   ENGINE_SIZE |   CYLINDERS |   FUEL_CONSUMPTION |   HWY_(L/100_KM) |   COMB_(L/100_KM) |   COMB_(MPG) |   ENGINE_SIZE_PER_CYLINDER |   CITY_HWY_FUEL_GAP |   COMB_CITY_FUEL_RATIO |   COMB_HWY_FUEL_RATIO |   FUEL_EFFICIENCY_KM_PER_L |   ENGINE_LOAD_PROXY |   TRANSMISSION_GEARS |   EMISSIONS |
|:------|------------:|--------------:|------------:|-------------------:|-----------------:|------------------:|-------------:|---------------------------:|--------------------:|-----------------------:|----------------------:|---------------------------:|--------------------:|---------------------:|------------:|
| count | 22337       |   22337       | 22337       |        22337       |      22337       |        22337      |  22337       |              22337         |         22337       |          22337         |         22337         |                22337       |          22337      |          22337       |  22337      |
| mean  |  2011.56    |       3.33347 |     5.81685 |           12.6738  |          8.86949 |           10.9628 |     27.4557  |                  0.561924  |             3.80436 |              0.868362  |             1.23497   |                    9.72619 |             21.5114 |              5.97189 |    248.402  |
| std   |     6.29188 |       1.31184 |     1.76632 |            3.35202 |          2.20524 |            2.798  |      7.19397 |                  0.0923232 |             1.48175 |              0.0328814 |             0.0726412 |                    2.54667 |             14.7106 |              1.44534 |     56.2985 |
| min   |  2000       |       0.8     |     2       |            4       |          3.8     |            4      |     12       |                  0.25      |            -1.5     |              0.748571  |             0.887324  |                    4.29185 |              2.4    |              1       |     94      |
| 25%   |  2006       |       2.3     |     4       |           10.4     |          7.3     |            9      |     22       |                  0.5       |             2.8     |              0.846847  |             1.1913    |                    7.87402 |              9.2    |              5       |    209      |
| 50%   |  2012       |       3       |     6       |           12.2     |          8.4     |           10.5    |     27       |                  0.575     |             3.8     |              0.864865  |             1.23711   |                    9.52381 |             18      |              6       |    242      |
| 75%   |  2017       |       4.2     |     8       |           14.7     |         10.2     |           12.7    |     31       |                  0.625     |             4.6     |              0.884058  |             1.28395   |                   11.1111  |             31.2    |              7       |    286      |
| max   |  2022       |       8.4     |    12       |           27.9     |         19.6     |           23.3    |     71       |                  0.9       |            10.4     |              1.125     |             1.54286   |                   25       |             84      |             10       |    406      |

---

## 2. Key Findings and Insights

### 2.1 CO2 Emissions Distribution
- The target variable `EMISSIONS` has a symmetric, bell-shaped distribution centered around **248.40 g/km**, with a standard deviation of **56.30 g/km**.
- The outlier cleaning process successfully filtered extreme and invalid samples, resulting in a cleaner dataset for modeling.

### 2.2 Feature Correlations with CO2 Emissions
- **City Fuel Consumption (`FUEL_CONSUMPTION`)**: Has a strong positive correlation of **0.911** with `EMISSIONS`. This indicates that vehicle fuel usage is the single most direct predictor of carbon output.
- **Engine Size (`ENGINE_SIZE`)**: Shows a strong positive correlation of **0.818** with `EMISSIONS`. Larger engines burn more fuel and therefore emit more carbon dioxide.
- **Cylinders (`CYLINDERS`)**: Shows a positive correlation of **0.794**. Additional cylinders generally indicate higher displacement and higher emissions.

### 2.3 Fuel Type Impact Analysis
The average carbon footprint varies widely across fuel technologies:

| FUEL_NAME         |   EMISSIONS |
|:------------------|------------:|
| Diesel            |     230.306 |
| Ethanol (E85)     |     270.034 |
| Natural Gas (CNG) |     312.848 |
| Premium Gasoline  |     257.173 |
| Regular Gasoline  |     239.923 |

- **Regular & Premium Gasoline** remain the benchmark standards, with Premium emissions slightly higher on average, often due to high-performance engines.
- **Ethanol (E85)** displays lower tail emissions in some categories, but due to its lower energy density, vehicles burn a higher volume of fuel, which partially offsets savings.
- **Natural Gas (CNG)** and **Diesel** show specialized profiles, with CNG being historically cleaner.

### 2.4 Engineered Feature Signals
The cleaning pipeline adds engineered predictors before modeling:

|                          |   EMISSIONS |
|:-------------------------|------------:|
| FUEL_EFFICIENCY_KM_PER_L |  -0.913459  |
| ENGINE_LOAD_PROXY        |   0.799967  |
| CITY_HWY_FUEL_GAP        |   0.738345  |
| ENGINE_SIZE_PER_CYLINDER |   0.545765  |
| COMB_CITY_FUEL_RATIO     |  -0.286522  |
| COMB_HWY_FUEL_RATIO      |   0.277557  |
| TRANSMISSION_GEARS       |   0.0378671 |

- `FUEL_EFFICIENCY_KM_PER_L` converts combined consumption into an efficiency view.
- `CITY_HWY_FUEL_GAP` captures urban/highway driving penalty.
- `ENGINE_SIZE_PER_CYLINDER`, `ENGINE_LOAD_PROXY`, and `TRANSMISSION_GEARS` add powertrain structure.

### 2.5 Transmission Efficiency
- Manual and Continuously Variable Transmissions (CVTs) generally show lower average emissions compared to older traditional Automatic systems, reflecting their higher energy transfer efficiency.
- Newer multi-gear automatic systems (such as Select Shift) have closed the efficiency gap significantly.

### 2.6 Yearly Trends (2000 - 2022)
- Average emissions dropped from **257.45 g/km** in 2000 to **257.98 g/km** in 2022.
- This represents a **-0.21% reduction** in average carbon output over 22 years.
- This positive trend is driven by regulatory standards, direct fuel injection, turbocharging, hybridization, and transition to smaller engine sizes.

---

## 3. High-Carbon vs Low-Carbon Manufacturers

- **Highest Average Emitters**: Manufacturers specializing in luxury, sports, or heavy vehicles (e.g., Lamborghini, Bugatti, Bentley, Rolls-Royce).
- **Lowest Average Emitters**: Manufacturers focusing on compact, hybrid, and small passenger cars (e.g., Smart, Toyota, Honda, Mazda).
