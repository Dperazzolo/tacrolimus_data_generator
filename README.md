# tacrolimus_data_generator (v1.0.0)

This repository contains the synthetic data generation code and resources associated with the study published on [ArXiv - Uncovering Population PK Covariates from VAE-Generated Latent Spaces](http://arxiv.org/abs/2505.02514)
---
This repository provides 2 generated example datasets (./example_generated_dataset folder) and a synthetic dataset generator for the pharmacokinetics (PK) of tacrolimus, built using a one-compartment population pharmacokinetic (popPK) model with first-order absorption and elimination. The model incorporates between-subject variability and covariate effects on clearance and volume of distribution. This framework is delivered through a Streamlit web application and is intended for researchers, pharmacometricians, and data scientists who need realistic PK simulations for analysis, model training, or exploration.

## Citation
---
```
@misc{perazzolo2025uncoveringpopulationpkcovariates,
      title={Uncovering Population PK Covariates from VAE-Generated Latent Spaces}, 
      author={Diego Perazzolo and Chiara Castellani and Enrico Grisan},
      year={2025},
      eprint={2505.02514},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.02514}, 
}
```
---

## Repository Purpose

You are allowed to:

- Download .csv files of generated dataset of Tacrolimus dosing and related covariates for analysis
- Generate realistic PK profiles of tacrolimus.
- Simulate various sampling designs (e.g., 48h, 120h)
- Provide structured input data for PK modeling and analysis
- Enable the exploration of on compartment model dosing and sampling scenarios

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

## Example Datasets

Pre-generated datasets are available in the [`example_generated_dataset`](./example_generated_dataset) folder:

- `tacrolimus_datasets_48h.zip`: Simulated 48-hour profiles
- `tacrolimus_datasets_120h.zip`: Simulated 120-hour profiles

Each ZIP archive contains:

- A CSV file with the **simulated covariates** for each virtual subject
- A CSV file with the corresponding **Tacrolimus concentration–time profiles**, generated using a one-compartment PK model with first-order absorption and elimination
---

## ⚠️ Disclaimer

The simulated data are for research and development purposes only. **They must not be used for clinical or diagnostic purposes.**

If you use this tool in your research or publication, please consider citing the paper (cite also this project and link).

---


