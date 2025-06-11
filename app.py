import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import streamlit as st

# Set beige background color using custom CSS
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5dc;
        }
        .stApp {
            background-color: #f5f5dc;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Load the trained pipeline
pipeline = joblib.load('student_pass_fail_model.pkl')

st.title("🎓 Student Pass/Fail Predictor App")
st.markdown("This app predicts whether a student will pass or fail and explains the prediction using LIME.")

# --------- Input Form ----------
with st.form("input_form"):
    gender = st.selectbox("👤 Gender", ["female", "male"])
    ethnicity = st.selectbox("🌎 Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    education = st.selectbox("🎓 Parental Education", [
        "some high school", "high school", "some college",
        "associate's degree", "bachelor's degree", "master's degree"
    ])
    lunch = st.selectbox("🍱 Lunch Type", ["standard", "free/reduced"])
    prep = st.selectbox("📘 Test Preparation", ["completed", "none"])
    math = st.slider("🧮 Math Score", 0, 100, 70)
    reading = st.slider("📖 Reading Score", 0, 100, 70)
    writing = st.slider("✍️ Writing Score", 0, 100, 70)
    submitted = st.form_submit_button("🔍 Predict")

# --------- Prediction and Explanation ----------
if submitted:
    preprocessor = pipeline.named_steps['preprocess']
    scaler = pipeline.named_steps['scale']
    model = pipeline.named_steps['model']

    # Input DataFrame
    input_df = pd.DataFrame({
        'gender': [gender],
        'race/ethnicity': [ethnicity],
        'parental level of education': [education],
        'lunch': [lunch],
        'test preparation course': [prep],
        'math score': [math],
        'reading score': [reading],
        'writing score': [writing]
    })

    # Make prediction
    prediction = pipeline.predict(input_df)[0]
    col1,col2,col3=st.columns([1,2,1])
    with col2:
        if prediction == 1:
            st.image("pass.gif")  # simple and clean
        else:
            st.image("fail.gif")

    pred_proba = pipeline.predict_proba(input_df)[0]
    result = "✅ Pass" if prediction == 1 else "❌ Fail"
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.subheader(f"🎓 Prediction: {result}")
        st.write(f"Confidence: **{pred_proba[prediction]*100:.2f}%**")


    

    # --------- Score Visualization ----------
    st.subheader("📊 Input Score Breakdown")
    st.markdown("""
**📌 What is shown in the Input Score Breakdown?**

This bar chart displays the student's individual scores in **Math**, **Reading**, and **Writing**.

- It helps visualize the strengths and weaknesses in their academic performance.
- Scores are on a scale from **0 to 100**.
- Higher bars indicate stronger performance in that subject.

This breakdown is useful context before interpreting the prediction and the LIME explanation.
""")

    fig, ax = plt.subplots()
    sns.barplot(x=['Math', 'Reading', 'Writing'], y=[math, reading, writing], palette="Blues_d", ax=ax)
    ax.set_ylim(0, 100)
    st.pyplot(fig)

    # --------- LIME Explanation ----------
    st.subheader("🔍 LIME Explanation")
    st.markdown("""
**ℹ️ What does the LIME chart show?**

The LIME (Local Interpretable Model-Agnostic Explanations) chart below explains which features most influenced the model's decision for this individual prediction.

- **Green bars** push the prediction towards **"Pass"**
- **Red bars** push the prediction towards **"Fail"**
- The **length of each bar** shows the strength of the impact

This helps you understand **why** the model predicted "Pass" or "Fail" for the student.
""")


    # Simulate background data by adding small noise to input
    # Create background data by copying and slightly perturbing only numeric columns
    background_raw = pd.concat([input_df] * 50, ignore_index=True)

# Add small noise ONLY to numeric columns (scores)
    numeric_cols = ['math score', 'reading score', 'writing score']
    background_raw[numeric_cols] = background_raw[numeric_cols] + np.random.normal(0, 1, background_raw[numeric_cols].shape)



    background_processed = preprocessor.transform(background_raw)
    background_scaled = scaler.transform(background_processed)

    feature_names = preprocessor.get_feature_names_out(input_df.columns)
    explainer = LimeTabularExplainer(
        training_data=background_scaled,
        feature_names=feature_names,
        class_names=['Fail', 'Pass'],
        mode='classification'
    )

    input_processed = preprocessor.transform(input_df)
    input_scaled = scaler.transform(input_processed)

    exp = explainer.explain_instance(
        data_row=input_scaled[0],
        predict_fn=model.predict_proba,
        num_features=10
    )

    # Create custom bar chart for explanation
    lime_explanation = exp.as_list()
    features = [x[0] for x in lime_explanation]
    values = [x[1] for x in lime_explanation]
    colors = ['green' if val > 0 else 'red' for val in values]

    fig, ax = plt.subplots()
    ax.barh(features, values, color=colors)
    ax.set_title("Top Features Influencing Prediction")
    ax.set_xlabel("Impact on Prediction (toward Pass)")
    ax.invert_yaxis()
    st.pyplot(fig)
