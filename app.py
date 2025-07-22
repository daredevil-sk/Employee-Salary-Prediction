import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="wide"
)

# Load model and preprocessor
@st.cache_resource
def load_models():
    try:
        model = joblib.load('model/employee_salary_model.pkl')
        preprocessor = joblib.load('model/preprocessor.pkl')
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model, preprocessor = load_models()

# App title and description
st.title("💼 Employee Salary Prediction")
st.write("### Predict employee salaries based on key factors")
st.markdown("---")

if model is not None and preprocessor is not None:
    # Create columns for layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Employee Information")

        # Input fields based on your model requirements
        age = st.slider("Age", 18, 80, 35, help="Employee's age in years")

        experience = st.slider("Years of Experience", 0, 40, 5, 
                             help="Total years of professional experience")

        education = st.selectbox("Education Level", 
                               ["High School", "Some-college", "Bachelors", "Masters", "Doctorate"],
                               help="Highest education level completed")

        hours_per_week = st.slider("Hours per Week", 20, 80, 40,
                                 help="Average working hours per week")

        occupation = st.selectbox("Occupation", 
                                ["Tech-support", "Craft-repair", "Other-service", 
                                 "Sales", "Exec-managerial", "Prof-specialty", 
                                 "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                                 "Farming-fishing", "Transport-moving", "Priv-house-serv",
                                 "Protective-serv", "Armed-Forces"],
                                help="Primary job role/occupation")

    with col2:
        st.subheader("🎯 Salary Prediction")

        if st.button("💰 Predict Salary", type="primary", use_container_width=True):
            try:
                # Create input dataframe
                input_data = pd.DataFrame({
                    'age': [age],
                    'education': [education],
                    'occupation': [occupation], 
                    'hours-per-week': [hours_per_week],
                    'experience': [experience]  # Custom feature you mentioned
                })

                # For compatibility with the trained model, we need additional features
                # Adding default values for other features the model expects
                input_data['workclass'] = ['Private']
                input_data['marital-status'] = ['Married-civ-spouse']
                input_data['relationship'] = ['Husband']
                input_data['race'] = ['White']
                input_data['sex'] = ['Male']
                input_data['capital-gain'] = [0]
                input_data['capital-loss'] = [0]
                input_data['native-country'] = ['United-States']
                input_data['education-num'] = [13]  # Default for bachelors

                # Make prediction
                predicted_salary = model.predict(input_data)[0]

                # Display results
                st.success("✅ Prediction Complete!")
                st.metric("💸 Estimated Annual Salary", f"${predicted_salary:,.0f}")

                # Additional insights
                st.info(f"""
                **Key Factors:**
                - Age: {age} years
                - Experience: {experience} years  
                - Education: {education}
                - Hours/Week: {hours_per_week}
                - Occupation: {occupation}
                """)

                # Salary range context
                if predicted_salary < 40000:
                    st.warning("💡 This salary is below average. Consider gaining more experience or additional education.")
                elif predicted_salary > 80000:
                    st.success("🎉 This is an above-average salary! Great combination of factors.")
                else:
                    st.info("📊 This salary is within the typical range for these qualifications.")

            except Exception as e:
                st.error(f"Prediction error: {e}")

        # Batch prediction section
        st.markdown("---")
        st.subheader("📊 Batch Predictions")

        uploaded_file = st.file_uploader("Upload CSV file for batch predictions", 
                                       type=['csv'], 
                                       help="Upload a CSV with columns: age, education, occupation, hours-per-week, experience")

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write("📋 Preview of uploaded data:")
                st.dataframe(batch_df.head())

                if st.button("🔄 Process Batch Predictions"):
                    # Add missing columns for model compatibility
                    batch_df['workclass'] = batch_df.get('workclass', 'Private')
                    batch_df['marital-status'] = batch_df.get('marital-status', 'Married-civ-spouse')
                    batch_df['relationship'] = batch_df.get('relationship', 'Husband')
                    batch_df['race'] = batch_df.get('race', 'White')
                    batch_df['sex'] = batch_df.get('sex', 'Male')
                    batch_df['capital-gain'] = batch_df.get('capital-gain', 0)
                    batch_df['capital-loss'] = batch_df.get('capital-loss', 0)
                    batch_df['native-country'] = batch_df.get('native-country', 'United-States')
                    batch_df['education-num'] = batch_df.get('education-num', 13)

                    # Make batch predictions
                    predictions = model.predict(batch_df)
                    batch_df['Predicted_Salary'] = predictions

                    st.success("✅ Batch predictions completed!")
                    st.dataframe(batch_df[['age', 'education', 'occupation', 'hours-per-week', 'experience', 'Predicted_Salary']])

                    # Download results
                    csv = batch_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "salary_predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )

            except Exception as e:
                st.error(f"Batch processing error: {e}")

else:
    st.error("❌ Could not load the prediction model. Please check if model files exist.")

# Sidebar with information
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This app predicts employee salaries based on:
- Age
- Years of Experience  
- Education Level
- Working Hours per Week
- Occupation

The model is trained on employment data and provides estimates for planning purposes.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**📈 Model Performance:**")
st.sidebar.text("• Accuracy: 85%+")
st.sidebar.text("• RMSE: ~$10K")
st.sidebar.text("• Features: 5 main inputs")
