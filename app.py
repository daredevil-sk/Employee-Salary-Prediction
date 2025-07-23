import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from io import StringIO

# Configure page
st.set_page_config(
    page_title="💼 Employee Salary Classification",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Load the best model (XGBoost) with error handling"""
    try:
        if os.path.exists('best_model.pkl'):
            model = joblib.load('best_model.pkl')
            return model, None
        else:
            return None, "❌ Model file 'best_model.pkl' not found!"
    except Exception as e:
        return None, f"❌ Error loading model: {str(e)}"

def validate_input_data(df):
    """Validate uploaded CSV data structure"""
    required_columns = ['age', 'education', 'occupation', 'hours-per-week']

    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        return False, f"Missing required columns: {missing_cols}"

    # Check data types and ranges
    if not pd.api.types.is_numeric_dtype(df['age']):
        return False, "Age column must be numeric"

    if not pd.api.types.is_numeric_dtype(df['hours-per-week']):
        return False, "Hours-per-week column must be numeric"

    # Check reasonable ranges
    if (df['age'] < 16).any() or (df['age'] > 90).any():
        return False, "Age values should be between 16 and 90"

    if (df['hours-per-week'] < 1).any() or (df['hours-per-week'] > 99).any():
        return False, "Hours per week should be between 1 and 99"

    return True, "Data validation passed"

def create_sample_data():
    """Create sample data for download"""
    sample_data = pd.DataFrame({
        'age': [25, 35, 45, 28, 52],
        'education': ['Bachelors', 'Masters', 'HS-grad', 'Bachelors', 'Doctorate'],
        'occupation': ['Tech-support', 'Exec-managerial', 'Craft-repair', 'Prof-specialty', 'Prof-specialty'],
        'hours-per-week': [40, 45, 38, 42, 35]
    })
    return sample_data

def format_prediction_results(predictions, probabilities):
    """Format prediction results for display"""
    results = []
    for pred, prob in zip(predictions, probabilities):
        salary_class = ">50K" if pred == 1 else "<=50K"
        confidence = prob if pred == 1 else (1 - prob)
        results.append({
            'Predicted_Salary_Class': salary_class,
            'Confidence': f"{confidence:.2%}",
            'Probability_High_Income': f"{prob:.2%}"
        })
    return pd.DataFrame(results)

def main():
    # Header
    st.title("💼 Employee Salary Classification")
    st.markdown("### Predict salary class (<=50K or >50K) using XGBoost")

    # Load model
    model, error_msg = load_model()

    if model is None:
        st.error(error_msg)
        st.info("""**To fix this issue:** Make sure 'best_model.pkl' is in your app directory""")
        return

    st.success("✅ XGBoost model loaded successfully!")

    # Sidebar for prediction type selection
    st.sidebar.header("🎯 Prediction Mode")
    prediction_type = st.sidebar.radio(
        "Choose prediction type:",
        ["Individual Prediction", "Batch Prediction"]
    )

    if prediction_type == "Individual Prediction":
        st.header("👤 Individual Salary Prediction")

        # Create input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                age = st.slider(
                    "Age", min_value=16, max_value=90, value=35,
                    help="Employee's age in years"
                )

                education = st.selectbox(
                    "Education Level",
                    options=[
                        'HS-grad', 'Some-college', 'Bachelors', 'Masters',
                        'Assoc-acdm', 'Assoc-voc', '11th', '9th', '7th-8th',
                        '12th', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'
                    ],
                    index=2,  # Default to Bachelors
                    help="Highest level of education achieved"
                )

                hours_per_week = st.slider(
                    "Hours per Week", min_value=1, max_value=80, value=40,
                    help="Number of hours worked per week"
                )

            with col2:
                occupation = st.selectbox(
                    "Occupation",
                    options=[
                        'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                        'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                        'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                        'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
                    ],
                    index=0,  # Default to Tech-support
                    help="Employee's occupation/job role"
                )

                st.info(f"""
                **Input Summary:**
                - Age: {age} years
                - Education: {education}
                - Occupation: {occupation}
                - Hours/week: {hours_per_week}
                """)

            # Prediction button
            predict_button = st.form_submit_button("🎯 Predict Salary Class", use_container_width=True)

            if predict_button:
                input_data = pd.DataFrame({
                    'age': [age],
                    'education': [education],
                    'occupation': [occupation],
                    'hours-per-week': [hours_per_week]
                })

                try:
                    prediction = model.predict(input_data)[0]
                    probability = model.predict_proba(input_data)[0]

                    salary_class = ">50K" if prediction == 1 else "<=50K"
                    confidence = probability[1] if prediction == 1 else probability[0]

                    if prediction == 1:
                        st.success(f"💰 **Predicted Salary Class: {salary_class}**")
                        st.info(f"📊 Probability of earning >50K: {probability[1]:.2%}")
                    else:
                        st.info(f"💼 **Predicted Salary Class: {salary_class}**")
                        st.info(f"📊 Probability of earning <=50K: {probability[0]:.2%}")

                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")

    else:  # Batch Prediction
        st.header("📁 Batch Salary Prediction")

        # Instructions
        with st.expander("📋 Instructions", expanded=True):
            st.markdown("""
            Prepare a CSV file with the required columns:
            - `age`
            - `education`
            - `occupation`
            - `hours-per-week`
            """)

        # Sample data download
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📥 Download Sample File")
            sample_data = create_sample_data()
            csv_sample = sample_data.to_csv(index=False)

            st.download_button(
                label="⬇️ Download Sample CSV",
                data=csv_sample,
                file_name="sample_employee_data.csv",
                mime="text/csv",
                help="Download a sample CSV file with the correct format"
            )

            st.dataframe(sample_data, use_container_width=True)

        with col2:
            st.subheader("📤 Upload Your Data")
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type=['csv'],
                help="Upload a CSV file with employee data for batch prediction"
            )

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ File uploaded successfully! ({len(df)} records)")

                    is_valid, validation_msg = validate_input_data(df)

                    if not is_valid:
                        st.error(f"❌ Data validation failed: {validation_msg}")
                        return

                    st.info(f"✅ {validation_msg}")

                    if st.button("🚀 Run Batch Predictions", use_container_width=True):
                        with st.spinner("Making predictions..."):
                            predictions = model.predict(df)
                            probabilities = model.predict_proba(df)[:, 1]  # Probability of >50K

                            results_df = format_prediction_results(predictions, probabilities)

                            final_results = pd.concat([df, results_df], axis=1)

                            st.subheader("📊 Prediction Results")
                            st.dataframe(final_results, use_container_width=True)

                            csv_results = final_results.to_csv(index=False)
                            st.download_button(
                                label="⬇️ Download Results CSV",
                                data=csv_results,
                                file_name="salary_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")

if __name__ == "__main__":
    main()
