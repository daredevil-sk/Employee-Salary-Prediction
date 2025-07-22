# Employee Salary Prediction Streamlit App

A machine learning web application for predicting employee salaries based on key factors.

## Features
- Individual salary predictions
- Batch prediction from CSV files
- Interactive web interface
- Model performance metrics

## Required Input Features
- Age (18-80 years)
- Years of Experience (0-40 years)
- Education Level (High School to Doctorate)
- Hours per Week (20-80 hours)
- Occupation (14+ categories)

## Local Deployment

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Extract the app folder
2. Navigate to the app directory:
   ```bash
   cd employee_salary_app
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Streamlit Cloud Deployment

### Step 1: Upload to GitHub
1. Create a new repository on GitHub
2. Upload all files from this folder to the repository
3. Make sure the repository is public or you have Streamlit Cloud access

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set the main file path: `app.py`
6. Click "Deploy"

### Step 3: Configuration
- **Repository:** your-username/your-repo-name
- **Branch:** main
- **Main file path:** app.py
- **Python version:** 3.9

### Step 4: Environment Variables (if needed)
No special environment variables required for this app.

## File Structure
```
employee_salary_app/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
└── model/
    ├── employee_salary_model.pkl  # Trained salary prediction model
    └── preprocessor.pkl           # Data preprocessing pipeline
```

## Model Information
- **Algorithm:** LightGBM Regressor
- **Preprocessing:** StandardScaler + OneHotEncoder
- **Performance:** ~85% accuracy, RMSE ~$10K
- **Training Data:** Adult/Census Income dataset

## Usage Examples

### CSV Format for Batch Predictions
```csv
age,education,occupation,hours-per-week,experience
35,Bachelors,Prof-specialty,40,10
28,Masters,Tech-support,45,5
42,High School,Craft-repair,40,15
```

## Troubleshooting
- Ensure all required files are present
- Check Python version compatibility
- Verify internet connection for initial package installation
- For Streamlit Cloud: check repository permissions and file paths

## Support
For issues with deployment, check:
1. Streamlit Cloud documentation
2. Repository file structure
3. Requirements.txt dependencies
