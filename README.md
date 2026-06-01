# Final - Data Science

This repository contains the code and datasets for a data science project analyzing traffic or congestion data (specifically relating to Vinh Tuy). The project workflow is organized into three main tasks spanning data processing, supervised learning, and unsupervised learning.

## Directory Structure

### `Code/`
Contains the Jupyter notebooks and Python scripts used for the analysis, divided into three tasks:
- **Task 1 (Data Processing & Preparation):** Includes data scraping (`DataScraping/Data_Scraping.py`), data cleaning, and exploratory data analysis (`Data_Processing_And_Preparation.ipynb`). Generates visualization plots such as target distribution, weather impacts, and correlation heatmaps.
- **Task 2 (Supervised Learning):** Contains predictive modeling using supervised techniques (`Supervised_Learning.ipynb`). Generates evaluation plots like confusion matrices, ROC curves, and feature importance.
- **Task 3 (Unsupervised Learning):** Focuses on clustering and pattern discovery (`Unsupervised_Learning.ipynb`). Outputs visualizations for optimal K selection, K-means PCA clusters, and cluster profile heatmaps.

### `Datasets/`
Contains the data files used and generated throughout the pipeline:
- `VINH_TUY.csv`: The raw initial dataset.
- `Cleaned Dataset.csv`: Data after the initial cleaning phase.
- `Processed Dataset.csv`: The finalized dataset ready for modeling.
- `Train Dataset.csv` & `Test Dataset.csv`: The train-test splits used for the supervised learning tasks.
