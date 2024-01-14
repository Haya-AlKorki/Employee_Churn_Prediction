import streamlit as st
import pandas as pd
import joblib
import subprocess
subprocess.call(['pip', 'install', 'joblib==1.3.2'])

gb_model = joblib.load('gradient_boosting_model.joblib')


st.title("Employee Churn Predicition | Machine Learning")


satisfaction_level = st.slider("Employee satisfaction level", 0.0, 1.0, 0.5)
last_evaluation = st.slider("Last evaluation score", 0.0, 1.0, 0.5)
number_projects = st.slider("Number of projects assigned to", 1, 10, 5)
average_monthly_hours = st.slider("Average monthly hours worked", 50, 300, 150)
time_spent_company = st.slider("Time spent at the company", 1, 10, 3)
work_accident = st.radio("Whether they have had a work accident", [0, 1])
promotion_last_5years = st.radio("Whether they have had a promotion in the last 5 years", [0, 1])

department_options = ('sales', 'technical', 'support', 'IT', 'product_mng', 'marketing', 'RandD', 'accounting',
                      'hr', 'management')
department = st.selectbox("Department name", department_options)

salary_options = ('low', 'medium', 'high')
salary = st.selectbox("Salary category", salary_options)

def predict_func():
        p1 = float(satisfaction_level)
        p2 = float(last_evaluation)
        p3 = float(number_projects)
        p4 = float(average_monthly_hours)
        p5 = float(time_spent_company)
        p6 = float(work_accident)
        p7 = float(promotion_last_5years)
        p8 = str(department)
        p9 = str(salary)

        p9_label_encoded = pd.Categorical([p9], categories=['high', 'low', 'medium']).codes[0]

        department_data = pd.DataFrame(0, index=[0], columns=department_options)
        department_data.loc[0, p8] = 1

        df = pd.DataFrame({
            'satisfaction_level': [p1],
            'last_evaluation': [p2],
            'number_project': [p3],
            'average_monthly_hours': [p4],
            'time_spend_company': [p5],
            'Work_accident': [p6],
            'promotion_last_5years': [p7],
            'salary': [p9_label_encoded],
            'Department_RandD': [department_data['RandD'][0]],
            'Department_accounting': [department_data['accounting'][0]],
            'Department_hr': [department_data['hr'][0]],
            'Department_management': [department_data['management'][0]],
            'Department_marketing': [department_data['marketing'][0]],
            'Department_product_mng': [department_data['product_mng'][0]],
            'Department_sales': [department_data['sales'][0]],
            'Department_support': [department_data['support'][0]],
            'Department_technical': [department_data['technical'][0]],
            'Department_IT': [department_data['IT'][0]]
        })


        column_order = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours',
                        'time_spend_company', 'Work_accident', 'promotion_last_5years', 'salary',
        'Department_IT', 'Department_RandD', 'Department_accounting', 'Department_hr',
        'Department_management', 'Department_marketing', 'Department_product_mng',
        'Department_sales', 'Department_support', 'Department_technical'] #+ list(department_options)
        df = df[column_order]
        prediction = gb_model.predict(df)

        st.write("User's inputs:")
        st.write(df)

        st.write("Predicition output:")
        if prediction == 1:
            st.write("Most probably the employee will churn.")
        else:
            st.write("Most probably the employee will retian.")

if st.button("Press the button to predict"):
    predict_func()
