# California Housing Price Prediction

A machine learning project that predicts housing prices in California using Linear Regression. This project uses the California Housing dataset from scikit-learn and implements a complete ML pipeline including data preprocessing, exploratory data analysis, model training, and evaluation.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Results](#results)

## Overview

This project demonstrates a complete machine learning workflow for predicting median house values in California districts. The model uses various features such as median income, house age, average rooms, and geographical location to make predictions.

## Dataset

The project uses the **California Housing Dataset** from scikit-learn, which contains information from the 1990 California census. The dataset includes:

- **20,640 samples**
- **8 features**
- **1 target variable** (median house value)

### Features

1. **MedInc** - Median income in block group
2. **HouseAge** - Median house age in block group
3. **AveRooms** - Average number of rooms per household
4. **AveBedrms** - Average number of bedrooms per household
5. **Population** - Block group population
6. **AveOccup** - Average number of household members
7. **Latitude** - Block group latitude
8. **Longitude** - Block group longitude

### Target Variable

- **Price** - Median house value (in $100,000s)

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Required Libraries

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Project

1. Clone the repository:
```bash
git clone <repository-url>
cd california-housing-prediction
```

2. Run the main script:
```bash
python house_price.py
```

3. The script will:
   - Load and prepare the data
   - Perform exploratory data analysis
   - Train a Linear Regression model
   - Evaluate the model performance
   - Save the trained model as `model.pkl`

### Making Predictions

To use the saved model for predictions:

```python
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model
model = pickle.load(open("model.pkl", "rb"))

# Prepare your data (make sure to normalize using the same scaler)
# Make prediction
prediction = model.predict(scaled_data)
```

## Project Structure

```
california-housing-prediction/
│
├── house_price.py          # Main script with complete pipeline
├── model.pkl              # Saved trained model
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## Model Performance

The Linear Regression model achieves the following performance metrics:

- **Mean Squared Error (MSE)**: Measures average squared difference between predicted and actual values
- **Mean Absolute Error (MAE)**: Average absolute difference between predictions and actual values
- **R² Score**: Indicates how well the model explains variance in the data
- **Root Mean Squared Error (RMSE)**: Square root of MSE, in same units as target variable
- **Adjusted R² Score**: R² score adjusted for number of features

### Model Pipeline

1. **Data Loading**: Fetch California Housing dataset
2. **Data Preparation**: Create DataFrame with features and target
3. **Exploratory Data Analysis**:
   - Correlation analysis
   - Pair plots for feature relationships
   - Box plots for outlier detection
4. **Data Preprocessing**:
   - Train-test split (70-30)
   - Feature normalization using StandardScaler
5. **Model Training**: Linear Regression
6. **Model Evaluation**: Multiple performance metrics
7. **Model Persistence**: Save model using pickle

## Technologies Used

- **Python 3.x** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning library
- **Seaborn** - Statistical data visualization
- **Matplotlib** - Plotting library
- **Pickle** - Model serialization

## Results

The project includes comprehensive visualizations:

- **Correlation heatmap** - Shows relationships between features
- **Pair plots** - Visualizes feature distributions and relationships
- **Box plots** - Identifies outliers in the dataset
- **Error distribution** - Shows prediction error patterns using KDE plot

The trained model is saved and can be reused for making predictions on new data without retraining.

## Future Improvements

- Implement additional regression algorithms (Random Forest, Gradient Boosting)
- Perform hyperparameter tuning
- Add cross-validation for more robust evaluation
- Create a web interface for user-friendly predictions
- Add feature engineering techniques
- Implement ensemble methods for improved accuracy

## License

This project is open source and available under the MIT License.

## Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This project is for educational purposes and demonstrates fundamental machine learning concepts using real-world data.
