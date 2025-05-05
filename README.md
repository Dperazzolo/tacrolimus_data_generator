 # tacrolimus_data_generator (v1.0.0)
The dataset generator and generated example dataset are related present in.... Please cite it. 
A synthetic dataset generator for the pharmacokinetics (PK) of tacrolimus, built using population pharmacokinetic (popPK) modeling and delivered through a Streamlit web application. This tool is intended for researchers, pharmacometricians, and data scientists who need realistic PK simulations for analysis, model training, or exploration.

---

## Project Purpose

This project allows you to:

- Generate realistic PK profiles of tacrolimus
- Simulate various sampling designs (e.g., 48h, 120h)
- Provide structured input data for PK modeling or AI/ML pipelines
- Enable the exploration of dosing and sampling scenarios

---

## Repository Structure

```text
tacrolimus_data_generator/
│
├── data_generator_application/
│   ├── pk_curve_generation.py               # Core logic for PK generation
│   └── popPK_tacrolimus_generator_app.py    # Streamlit interface
│
├── example_generated_dataset/
│   ├── tacrolimus_datasets_48h.zip          # Example dataset (48h)
│   └── tacrolimus_datasets_120h.zip         # Example dataset (120h)
│
├── README.md
└── LICENSE
```

---

## How to Install and Run the App

### 1. Clone the repository
```bash
git clone https://github.com/your_username/tacrolimus_data_generator.git
cd tacrolimus_data_generator
```

### 2. Install dependencies
It is highly recommended to use a virtual environment:
```bash
pip install streamlit pandas numpy
```

### 3. Launch the Streamlit app
```bash
streamlit run data_generator_application/popPK_tacrolimus_generator_app.py
```

---

## How to Generate a Dataset

Once the app is running in your browser:

1. Set the number of patients to simulate
2. Define the dosing regimen and PK parameters
3. Choose the sampling time window (e.g., 48h or 120h)
4. Click "Generate" to simulate the PK curves
5. Download the generated dataset as a `.csv` file

---
## ⚠️ Disclaimer

The simulated data are for research and development purposes only. **They must not be used for clinical or diagnostic purposes.**

If you use this tool in your research or publication, please consider citing this project or linking to the repository.

---


## Citation
......(to complete)......


