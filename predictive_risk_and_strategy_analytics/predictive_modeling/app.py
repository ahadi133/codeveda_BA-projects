# Run locally:  pip install streamlit scikit-learn joblib pandas   then   streamlit run app.py
# Prototype only - keep churn_model.joblib and house_model.joblib in the same folder.
import streamlit as st, pandas as pd, joblib

st.set_page_config(page_title="Codveda ML Prototype", layout="centered")
st.title("Codveda ML prototype")
tab1, tab2 = st.tabs(["Customer churn risk", "House price (Boston data)"])

with tab1:
    art = joblib.load("churn_model.joblib"); model, cols = art["model"], art["columns"]
    c1, c2 = st.columns(2)
    state = c1.text_input("State (2-letter)", "KS").upper()
    area = c2.selectbox("Area code", [408, 415, 510])
    intl = c1.checkbox("International plan"); vm = c2.checkbox("Voice mail plan")
    acct = c1.number_input("Account length", 1, 250, 100); vmsg = c2.number_input("Voicemail messages", 0, 60, 0)
    dm = c1.number_input("Day minutes", 0.0, 400.0, 180.0); dc = c2.number_input("Day calls", 0, 200, 100)
    em = c1.number_input("Evening minutes", 0.0, 400.0, 200.0); ec = c2.number_input("Evening calls", 0, 200, 100)
    nm = c1.number_input("Night minutes", 0.0, 400.0, 200.0); nc = c2.number_input("Night calls", 0, 200, 100)
    im = c1.number_input("Intl minutes", 0.0, 25.0, 10.0); ic = c2.number_input("Intl calls", 0, 25, 4)
    svc = st.slider("Customer service calls", 0, 10, 1)
    if st.button("Predict churn risk"):
        row = {"Account length": acct, "International plan": int(intl), "Voice mail plan": int(vm), "Number vmail messages": vmsg,
               "Total day minutes": dm, "Total day calls": dc, "Total eve minutes": em, "Total eve calls": ec, "Total night minutes": nm,
               "Total night calls": nc, "Total intl minutes": im, "Total intl calls": ic, "Customer service calls": svc}
        X = pd.DataFrame([row]).reindex(columns=cols, fill_value=0)
        for k in (f"State_{state}", f"Area code_{area}"):
            if k in X.columns: X[k] = 1
        p = float(model.predict_proba(X)[0, 1]); band = "Low" if p < .3 else "Medium" if p < .6 else "High"
        st.metric("Predicted churn probability", f"{p:.1%}"); st.write(f"Risk band (illustrative thresholds): **{band}**")
        st.caption("Prototype. Probabilities are estimates, not certainties; human review required.")

with tab2:
    art = joblib.load("house_model.joblib"); hm, hcols = art["model"], art["columns"]
    vals = {}
    defaults = dict(CRIM=0.25, ZN=0.0, INDUS=9.0, CHAS=0, NOX=0.54, RM=6.3, AGE=68.0, DIS=3.8, RAD=5, TAX=330.0, PTRATIO=18.5, B=391.0, LSTAT=12.5)
    cc = st.columns(3)
    for i, f in enumerate(hcols): vals[f] = cc[i % 3].number_input(f, value=float(defaults[f]))
    if st.button("Predict median home value"):
        st.metric("Predicted MEDV (USD thousands)", f"{hm.predict(pd.DataFrame([vals]))[0]:.1f}")
        st.caption("Classic Boston Housing data (1970s) - demo only, not for real valuation.")
