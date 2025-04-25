import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pk_curve_generation import generate_person_samples, generate_pk_signals_tracolimus
from datetime import datetime
import io
import zipfile


# Get today's date in YYYY-MM-DD format
today = datetime.today().strftime('%Y-%m-%d')

st.set_page_config(layout="wide")

st.title("Tacrolimus PK Synthetic Dataset Generator")
st.markdown("Simulate covariates and concentration-time profiles for patients treated with Tacrolimus. Data generated are based on a one-compartment model.")



# --- User Inputs ---
n_samples = st.number_input("Number of subjects to simulate:", min_value=10, max_value=20000, value=100, step=10)
max_hour = st.number_input("Duration of simulation (hours):", min_value=6, max_value=240, value=48, step=1)
dose = st.number_input("Administered dose (mg):", min_value=1.0, max_value=1000.0, value=300.0, step=10.0)
seed = st.number_input("Random seed (for reproducibility):", min_value=0, value=42, step=1)

# Manual setup for constants
tlag = 0.346 # define manually the tlag (time between the administration and drug absorpion)
random_error = 0.1 # parameter that define inter-variability subjects.

# Generate data on button click
if st.button("Generate Dataset"):
    np.random.seed(seed)
    st.session_state["persons"] = generate_person_samples(n_samples)
    st.session_state["max_hour"] = max_hour
    st.session_state["dose"] = dose
    st.session_state["random_error"] = random_error
    st.session_state["tlag"] = tlag
    st.success("Subjects successfully generated!")

# Reset session state
if st.button("Reset"):
    st.session_state.clear()

# If data has been generated previously
if "persons" in st.session_state:
    persons = st.session_state["persons"]
    max_hour = st.session_state["max_hour"]
    dose = st.session_state["dose"]
    tlag = st.session_state["tlag"]
    random_error = st.session_state["random_error"]

    # Generate PK signal generator
    generator = generate_pk_signals_tracolimus(
        dose=dose,
        er_random=random_error,
        name="TacrolimusSim",
        td=0,
        tlag=tlag
    )

    conc_data = []
    covariate_data = []

    for i, person in enumerate(persons):
        hours, conc = generator.concetration_over_time_pk_model(person, hours_to_sample=int(max_hour + 1))
        conc_data.append([i] + conc)
        covariate_data.append({
            'ID': i,
            'age': person.age,
            'sex': person.sex,
            'race': person.race,
            'weight': person.weight,
            'SNP': person.SNP,
            'hemoglobine': person.hemoglobine,
            'albumine': person.albumine,
            'last_dose_time': person.last_dose_time,
            'mg_twice_daily_dose': person.mg_twice_daily_dose,
            'blood_conc': person.blood_conc,
            'CL': person.pk_params['CL'],
            'V': person.pk_params['V'],
            'ke': person.pk_params['ke']
        })

    df_cov = pd.DataFrame(covariate_data)
    df_conc = pd.DataFrame(conc_data, columns=["ID"] + [f"t{h}" for h in range(0, int(max_hour + 1))])

    st.subheader("📊 Sampled Covariates")
    st.dataframe(df_cov.head())

    st.subheader("🧪 Simulated PK Profiles")
    st.dataframe(df_conc.head())

    # Download button (zi file with both Pk Curves and Covariates)
    # Create in-memory zip file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        # Add covariates CSV
        zf.writestr(f"{today}_tacrolimus_covariates.csv", df_cov.to_csv(index=False))
        # Add PK profiles CSV
        zf.writestr(f"{today}_tacrolimus_pk_signals.csv", df_conc.to_csv(index=False))

    # Prepare download
    zip_buffer.seek(0)
    st.download_button(
        label="📥 Download ZIP of Both Datasets",
        data=zip_buffer,
        file_name="tacrolimus_datasets.zip",
        mime="application/zip"
)



    # Plot preview of PK profiles
    st.subheader("📈 Preview of 5 PK Profiles (randomly extracted)")
    fig, ax = plt.subplots(figsize=(15, 6))
    for i in df_conc.sample(min(5, len(df_conc))).itertuples():
        ax.plot(range(0, int(max_hour + 1)), i[2:], label=f"ID {i.ID}")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Concentration (ng/mL)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)