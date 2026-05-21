# Data Structure and Processing Pipeline

This document explains the workflow and transformations applied to transition from the raw data to the final processed dataset used for modeling and visualization.

## 1. Raw Data Collection
The raw data consists of traffic and environmental logs (e.g., `VINH_TUY.csv` in `Raw Datasets/`). These logs contain continuous time-series data detailing route speeds, travel times, weather conditions, and incident reports.

## 2. Data Cleaning (`Data_Cleaning.py`)
The first step is to sanitize the raw data:
- **Missing Values:** Rows with null or incomplete essential data are dropped to ensure data integrity.
- **Output:** The intermediate result is saved as `Cleaned Dataset.csv`.

## 3. Data Processing and Feature Engineering (`Data_Processing.py`)
This script takes `Cleaned Dataset.csv` and transforms it into the final machine-learning-ready structure (`Processed Dataset.csv`).

### A. Dropping Leakage and Redundant Columns
To prevent "data leakage" (where a model is trained on information it wouldn't have in a real-world prediction scenario) and reduce noise, we drop columns such as:
- `current_speed`, `speed_deficit`, `travel_time_s`, `free_flow_time_s`, `speed_limit_baseline`, `speed_ratio_proxy`
- `magnitude`, `incident_type`, `frc_class`, `is_weekend`, `route_name`, `timestamp`

*(Note: `route_delay_s` and `hour_of_day` are explicitly retained as they are critical for exploratory data visualization).*

### B. Feature Engineering
We create new, highly predictive features based on existing data:
- **`is_rush_hour`:** A binary feature set to `1` if the `hour_of_day` falls within peak commuting times (7 AM, 4 PM, 5 PM).
- **`weather` Simplification:** Raw weather strings are simplified into categorical buckets (`Clear`, `Rain`, `Overcast`, `Partially cloudy`).
- **`adverse_weather_score`:** A custom severity score (`0`, `1`, or `2`) calculated using `rain_mm` and `visibility`. For instance, heavy rain (>7.6mm) or extremely low visibility (<400m) yields a score of `2`.
- **`rush_weather_interaction`:** An interaction term multiplying `is_rush_hour` by `adverse_weather_score` to capture the compounded effect of bad weather during peak traffic.

### C. Categorical Encoding
Machine learning models require numerical inputs. We apply `LabelEncoder` to convert textual categorical columns (like `direction` and `weather`) into numerical integer IDs.

## 4. Final Output
The fully engineered pipeline exports to `Processed Datasets/Processed Dataset.csv`. This single, unified file is then consumed directly by our visualization scripts and supervised learning models.
