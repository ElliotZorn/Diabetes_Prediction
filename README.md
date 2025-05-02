# Diabetes Prediction using Multilayer Perceptron (MLP)

Early detection of diabetes is crucial but often challenging due to the absence of symptoms in its initial stages.  
This project aims to build a **Multilayer Perceptron (MLP)** model that can predict whether a patient is at risk of developing diabetes based on medical and lifestyle-related features.

## Technical Requirements

This project requires the following dependencies, which are managed through Conda:

- `python==3.9`
- `numpy==1.21`
- `pandas==1.3`
- `matplotlib==3.4`
- `keras==2.6`
- `opencv==4.5`
- `seaborn==0.11`
- `scikit-learn==0.24`
- `pillow==8.3`
- `ipykernel==6.0`
- `jupyter`
- `piexif==1.1.2` (via `pip`)

All dependencies are listed in the `environment.yml` file included in the project files.

## Dataset Description

The dataset contains the following attributes:

- `Pregnancies` 
- `Glucose`
- `BloodPressure` 
- `SkinThickness`
- `Insulin`
- `BMI`  
- `DiabetesPedigreeFunction` 
- `Age` 
- `Outcome`


## Setup Instructions

To set up the environment using Conda:

```bash
conda env create -f environment.yml
```

To activate the enviorment:

```bash
conda activate project3
```