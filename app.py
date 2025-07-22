import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Configure page
st.set_page_config(
    page_title="💼 Employee Salary Prediction",
    page_icon="💼",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model with error handling"""
    try:
        # Try loading the complete pipeline first
        if os.path.exists('employee_salary_model.pkl'):
            model = joblib.load('employee_salary_model.pkl')
            st.success("✅ Model loaded successfully!")
            return model
        else:
            st.error("❌ Model file 'employee_salary_model.pkl' not found!")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def main():
    st.title("💼 Employee Salary Prediction")
    st.markdown("### Predict employee salaries based on demographic and work factors")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("⚠️ Cannot proceed without a trained model. Please check your model files.")
        st.info("""
        **To fix this issue:**
        1. Run the model extraction script to create the .pkl files
        2. Make sure 'employee_salary_model.pkl' is in your app directory
        3. Redeploy your Streamlit app
        """)
        return
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Personal Information")
        age = st.slider("Age", 18, 80, 35)
        education = st.selectbox("Education Level", [
            'Bachelors', 'HS-grad', 'Some-college', 'Masters', 
            'Assoc-acdm', '11th', 'Assoc-voc', '9th', '7th-8th', 
            '12th', 'Doctorate', '5th-6th', '10th', '1st-4th', 'Preschool'
        ])
        race = st.selectbox("Race", [
            'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'
        ])
        sex = st.selectbox("Gender", ['Male', 'Female'])
        
    with col2:
        st.subheader("💼 Work Information")
        workclass = st.selectbox("Work Class", [
            'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
            'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
        ])
        occupation = st.selectbox("Occupation", [
            'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
            'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
            'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
            'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
        ])
        hours_per_week = st.slider("Hours per Week", 20, 80, 40)
        marital_status = st.selectbox("Marital Status", [
            'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
            'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
        ])
        
    # Additional features
    st.subheader("📊 Additional Information")
    col3, col4 = st.columns(2)
    
    with col3:
        relationship = st.selectbox("Relationship", [
            'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'
        ])
        native_country = st.selectbox("Native Country", [
            'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada',
            'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece',
            'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy',
            'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland',
            'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti',
            'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland',
            'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru',
            'Hong', 'Holand-Netherlands'
        ])
        
    with col4:
        education_num = st.slider("Education Years", 1, 16, 13)
        capital_gain = st.number_input("Capital Gain", 0, 99999, 0)
        capital_loss = st.number_input("Capital Loss", 0, 4356, 0)
        fnlwgt = st.number_input("Final Weight", 12285, 1484705, 200000)
    
    # Prediction
    if st.button("🎯 Predict Salary", type="primary", use_container_width=True):
        try:
            # Create input dataframe
            input_data = pd.DataFrame({
                'age': [age],
                'workclass': [workclass],
                'fnlwgt': [fnlwgt],
                'education': [education],
                'education-num': [education_num],
                'marital-status': [marital_status],
                'occupation': [occupation],
                'relationship': [relationship],
                'race': [race],
                'sex': [sex],
                'capital-gain': [capital_gain],
                'capital-loss': [capital_loss],
                'hours-per-week': [hours_per_week],
                'native-country': [native_country]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.success(f"💰 **Predicted Annual Salary: ${prediction:,.0f}**")
            
            # Show confidence intervals
            if prediction <= 40000:
                st.info("📊 This prediction suggests a lower income bracket")
            elif prediction <= 60000:
                st.info("📊 This prediction suggests a middle income bracket")
            else:
                st.info("📊 This prediction suggests a higher income bracket")
                
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Please check your input values and try again.")
    
    # Model info sidebar
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.info("""
        **Algorithm:** LightGBM Regressor
        
        **Features Used:**
        - Age, Education, Work Hours
        - Occupation, Work Class
        - Demographics
        - Capital Gains/Losses
        
        **Training Data:** Adult Census Dataset
        """)

if __name__ == "__main__":
    main()
