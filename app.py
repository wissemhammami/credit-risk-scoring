import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

from src.models.scorecard import create_scorecard
from src.models.registry import ModelRegistry

st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="🏦",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(BASE_DIR / "models" / "feature_pipeline.joblib")
    registry = ModelRegistry(base_path=BASE_DIR / "models_artifacts")
    model, _ = registry.load_latest_model()
    return pipeline, model


try:
    pipeline, model = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Failed to load model: {e}")

st.title("🏦 Credit Risk Scoring")
st.markdown("Enter applicant details to get a credit decision.")
st.divider()

with st.form("applicant_form"):

    st.subheader("Personal Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=70, value=35)
        gender = st.selectbox("Gender", ["M", "F", "XNA"])
        education = st.selectbox("Education", [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree",
        ])
        family_status = st.selectbox("Family Status", [
            "Married",
            "Single / not married",
            "Civil marriage",
            "Separated",
            "Widow",
        ])

    with col2:
        income = st.number_input(
            "Annual Income ($)", min_value=0, max_value=10000000, value=150000, step=5000
        )
        income_type = st.selectbox("Income Type", [
            "Working",
            "Commercial associate",
            "Pensioner",
            "State servant",
            "Unemployed",
        ])
        housing_type = st.selectbox("Housing Type", [
            "House / apartment",
            "With parents",
            "Municipal apartment",
            "Rented apartment",
            "Office apartment",
            "Co-op apartment",
        ])
        contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])

    st.subheader("Loan Details")
    col3, col4 = st.columns(2)

    with col3:
        credit_amount = st.number_input(
            "Credit Amount ($)", min_value=0, max_value=10000000, value=500000, step=10000
        )
        annuity = st.number_input(
            "Annual Annuity ($)", min_value=0, max_value=1000000, value=25000, step=1000
        )

    with col4:
        goods_price = st.number_input(
            "Goods Price ($)", min_value=0, max_value=10000000, value=450000, step=10000
        )
        days_employed = st.number_input(
            "Years Employed", min_value=0, max_value=50, value=5
        )

    st.subheader("External Scores")
    col5, col6, col7 = st.columns(3)
    with col5:
        ext1 = st.slider("External Score 1", 0.0, 1.0, 0.3)
    with col6:
        ext2 = st.slider("External Score 2", 0.0, 1.0, 0.3)
    with col7:
        ext3 = st.slider("External Score 3", 0.0, 1.0, 0.3)

    submitted = st.form_submit_button("Predict Credit Risk", use_container_width=True)

if submitted and model_loaded:

    input_data = pd.DataFrame([{
        "CODE_GENDER":                gender,
        "NAME_CONTRACT_TYPE":         contract_type,
        "NAME_EDUCATION_TYPE":        education,
        "NAME_FAMILY_STATUS":         family_status,
        "NAME_HOUSING_TYPE":          housing_type,
        "NAME_INCOME_TYPE":           income_type,
        "AMT_INCOME_TOTAL":           float(income),
        "AMT_CREDIT":                 float(credit_amount),
        "AMT_ANNUITY":                float(annuity),
        "AMT_GOODS_PRICE":            float(goods_price),
        "DAYS_BIRTH":                 float(-age * 365),
        "DAYS_EMPLOYED":              float(-days_employed * 365),
        "DAYS_REGISTRATION":          -1000.0,
        "DAYS_ID_PUBLISH":            -1000.0,
        "EXT_SOURCE_1":               float(ext1),
        "EXT_SOURCE_2":               float(ext2),
        "EXT_SOURCE_3":               float(ext3),
        "ORGANIZATION_TYPE":          "Business Entity Type 3",
        "WEEKDAY_APPR_PROCESS_START": "MONDAY",
        "NAME_TYPE_SUITE":            "Unaccompanied",
    }])

    try:
        # Get all columns the pipeline expects
        preprocessor = pipeline.named_steps["preprocessor"]
        all_expected = []
        for _, _, cols in preprocessor.transformers:
            if isinstance(cols, list):
                all_expected.extend(cols)

        # Fill missing columns all at once — avoids fragmentation warning
        missing_cols = [col for col in all_expected if col not in input_data.columns]
        missing_df = pd.DataFrame(0, index=input_data.index, columns=missing_cols)
        input_data = pd.concat([input_data, missing_df], axis=1)

        # Transform and predict
        X_transformed = pipeline.transform(input_data)
        pd_prob = float(model.predict_proba(X_transformed)[0][1])

        # Scorecard
        scorecard = create_scorecard(
            ids=pd.Series([1]),
            predictions=pd.Series([pd_prob]),
        )

        score = float(scorecard["Score"].iloc[0])
        decision = scorecard["Decision"].iloc[0]

        st.divider()
        st.subheader("Results")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Probability of Default", f"{pd_prob:.1%}")
        with col_b:
            st.metric("Credit Score", f"{score:.0f}")
        with col_c:
            if decision == "Approve":
                st.success(f"✅ {decision}")
            elif decision == "Reject":
                st.error(f"❌ {decision}")
            else:
                st.warning(f"⚠️ {decision}")

        st.divider()
        st.markdown("**Risk Level**")
        st.progress(float(pd_prob))

        if pd_prob < 0.1:
            st.markdown("🟢 Low Risk")
        elif pd_prob < 0.3:
            st.markdown("🟡 Medium Risk")
        else:
            st.markdown("🔴 High Risk")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)