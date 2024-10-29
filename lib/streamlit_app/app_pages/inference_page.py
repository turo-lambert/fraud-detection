"""Useful page to run inference on the model."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st

# URL for the REST API endpoint
# API_URL = "http://localhost:5001/score"
API_URL = "http://127.0.0.1:5001/score"


def write() -> None:
    """Writes the page."""
    st.title("Fraud Detection Model Inference")

    # Define placeholders for each payload field
    payload = {
        "sys_sector": st.selectbox("System Sector", ["Private NonLife"]),
        "sys_label": st.selectbox("System Label", ["FRISS"]),
        "sys_process": st.selectbox("System Process", ["Claims_initial_load"]),
        "sys_product": st.selectbox("System Product", ["MOTOR"]),
        "sys_dataspecification_version": st.selectbox("Data Specification Version", [4.5]),
        "sys_claimid": st.text_input("Claim ID", "MTR-405630423-02"),
        "sys_currency_code": st.selectbox("Currency Code", ["EUR"]),
        "claim_amount_claimed_total": st.number_input("Claim Amount Claimed Total", min_value=0, step=1, value=41),
        "claim_causetype": st.selectbox(
            "Claim Cause Type", ["Collision", "Other", "Theft", "Weather", "Animals", np.nan], index=0
        ),
        "claim_date_occurred": int(
            st.number_input("Claim Date Occurred (YYYYMMDD)", min_value=0, step=1, value=20120730)
        ),
        "claim_date_reported": int(
            st.number_input("Claim Date Reported (YYYYMMDD)", min_value=0, step=1, value=20121101)
        ),
        "claim_location_urban_area": st.selectbox("Claim Location Urban Area", [0, 1], index=0),
        "object_make": st.selectbox(
            "Object Make", ["VOLKSWAGEN", "CITROEN", "RENAULT", "BMW", "AUDI", "OTHER", "OPEL"], index=4
        ),
        "object_year_construction": st.selectbox(
            "Object Year of Construction",
            [
                2008.0,
                2003.0,
                2001.0,
                2017.0,
                2011.0,
                2007.0,
                2009.0,
                2013.0,
                2000.0,
                2010.0,
                2014.0,
                2015.0,
                2005.0,
                2004.0,
                2012.0,
                1996.0,
                2006.0,
                1997.0,
                1977.0,
                2002.0,
                1999.0,
                1998.0,
                2016.0,
                2018.0,
                1991.0,
                1988.0,
                1993.0,
                1995.0,
                1982.0,
                1987.0,
                1994.0,
                1992.0,
                1989.0,
                1975.0,
                1986.0,
                1984.0,
                1990.0,
                1974.0,
                2019.0,
                1969.0,
                2020.0,
                1985.0,
                1980.0,
                1968.0,
                1970.0,
                1983.0,
                1981.0,
                1979.0,
                1948.0,
            ],
            index=9,
        ),
        "ph_firstname": st.text_input("Policy Holder First Name", "Jamel"),
        "ph_gender": st.selectbox("Policy Holder Gender", ["F", "M", "L", np.nan], index=1),
        "ph_name": st.text_input("Policy Holder Name", "Henry"),
        "policy_fleet_flag": st.selectbox("Policy Fleet Flag", [0, 1], index=0),
        "policy_insured_amount": st.number_input("Policy Insured Amount", min_value=0, step=1, value=142286),
        "policy_profitability": st.selectbox(
            "Policy Profitability", ["Low", "Very low", "High", "Very high", "Neutral"], index=1
        ),
    }

    # Convert the payload to a format that the API expects
    formatted_payload = {key: [value] for key, value in payload.items()}

    # Button to send the request
    if st.button("Run Inference"):
        try:
            response = requests.post(API_URL, json=formatted_payload, timeout=5)
            if response.status_code == 200:
                prediction = response.json().get("prediction")
                st.success(f"Prediction: {prediction}")
                processed_obs = pd.DataFrame.from_dict(response.json().get("processed_obs"))
                shap_values = np.array(response.json().get("shap_values"))
                st.subheader("SHAP Feature Importance")
                fig, _ = plt.subplots()
                shap.summary_plot(
                    shap_values,
                    processed_obs,
                    plot_type="bar",
                    show=False,
                    title="Average impact on model output magnitude",
                    plot_size=(20, 5),
                )
                st.pyplot(fig, use_container_width=True)

            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
