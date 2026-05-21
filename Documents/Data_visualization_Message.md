# The "Message" of the Work

## What are we trying to achieve?
The central goal of this data science project is to **understand, predict, and ultimately mitigate traffic congestion**. By collecting raw data points (such as travel times, weather conditions, and incident reports), we aim to translate messy, real-world traffic dynamics into clear, actionable insights and machine learning models. 

Our core "message" is that congestion is not entirely random; it is driven by identifiable factors—such as time of day, severe weather, and their compounded interactions.

## Why did we process and engineer the data?
Raw traffic data contains noise and "data leakage" (variables that perfectly give away the answer, like `current_speed`, which a model wouldn't have access to in the future). 
We processed the data to:
1. Strip away unfair advantages (leakage) so our models learn the true underlying patterns.
2. Engineer human-interpretable features (like an `adverse_weather_score` and a `rush_weather_interaction` term) that bridge the gap between raw numbers and actual driver experiences.

## Why did we visualize these specific files?
Data visualization is how we prove our hypotheses before we ever train a model. Each graph we generated serves a distinct analytical purpose:

1. **Target Distribution (`1_target_distribution.png`)**: 
   - *Message:* Are traffic jams rare or common? This graph shows the balance between congested (`1`) and flowing (`0`) states. If the data is heavily skewed, we know our models will need special treatment for class imbalance.

2. **Route Delay Distribution (`2_route_delay_distribution.png`)**:
   - *Message:* When traffic slows down, how bad is it usually? This density histogram reveals whether delays are usually minor inconveniences or massive, long-tail gridlocks.

3. **Route Delay by Hour (`3_delay_by_hour.png`)**:
   - *Message:* Traffic is dictated by human schedules. This boxplot visually confirms when "rush hour" truly occurs, proving the validity of our engineered `is_rush_hour` feature.

4. **Temperature vs. Humidity by Congestion (`4_temp_vs_humidity_scatter.png`)**:
   - *Message:* Do environmental comfort factors play a role? By mapping temperature and humidity and coloring by congestion, we can visually scan for "danger zones" where specific weather conditions consistently lead to traffic jams.

5. **Correlation Heatmap (`5_correlation_heatmap.png`)**:
   - *Message:* Everything is connected, but some connections are stronger than others. The heatmap statistically verifies which features strongly influence each other and which have the highest correlation with our target (`is_congested`). This acts as the ultimate sanity check for our feature engineering phase.
