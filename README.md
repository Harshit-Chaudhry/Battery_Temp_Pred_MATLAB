

# 🔋 Battery_Temp_Pred_MATLAB

This MATLAB project predicts the battery surface temperature under varying duty cycles and conditions using modeling and image datasets.

---

## 📁 Repository Structure

```
Battery_Temp_Pred_MATLAB/
│
├── c1_15du_new_Images/           # Image data for Class 1, 15 DU
├── c1_15du_new_Models/           # Models for Class 1, 15 DU
├── c1_15du_newoth_Images/        # Other Image data for Class 1, 15 DU
├── c1_15du_newoth_Models/        # Other Models for Class 1, 15 DU
├── c1_35du_new_Images/           # Image data for Class 1, 35 DU
├── c1_35du_new_Models/           # Models for Class 1, 35 DU
├── c2_15du_new_Images/           # Image data for Class 2, 15 DU
├── c2_15du_new_Models/           # Models for Class 2, 15 DU
├── c3_15du_new_Images/           # Image data for Class 3, 15 DU
├── c3_15du_new_Models/           # Models for Class 3, 15 DU
├── 1c 15du new.xlsx              # Excel data for Class 1, 15 DU
├── 2c 15du new.xlsx              # Excel data for Class 2, 15 DU
├── 3c 15du new.xlsx              # Excel data for Class 3, 15 DU
├── 3c 35du new.xlsx              # Excel data for Class 3, 35 DU
├── c1_15du.m                     # MATLAB script for Class 1, 15 DU
├── c2_15du.m                     # MATLAB script for Class 2, 15 DU
├── c3_15du_new.m                 # MATLAB script for Class 3, 15 DU
├── c3_35du_new.m                 # MATLAB script for Class 3, 35 DU
└── oth_1c_35du.m                 # MATLAB script for Other conditions, Class 1, 35 DU
```

---

## 📌 Objective

To build predictive models for battery surface temperature using training datasets under different duty cycles and operating classes using MATLAB.

---

## 🚀 Getting Started

### Prerequisites
- MATLAB R2021b or later
- Curve Fitting and Image Processing Toolboxes (recommended)

### Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/Harshit-Chaudhry/Battery_Temp_Pred_MATLAB.git
   cd Battery_Temp_Pred_MATLAB
   ```

2. Open MATLAB.

3. Run any of the script files such as:
   ```matlab
   run('c1_15du.m')
   ```

---

## 📊 Sample Plot

Here’s a sample output graph showing temperature prediction:

![Prediction Graph](https://raw.githubusercontent.com/Harshit-Chaudhry/Battery_Temp_Pred_MATLAB/c3_15du_new_Images/model_comparison.png)

---

## 🧠 Model Info

- Data Source: Excel sheets & pre-labeled image datasets.
- Processing: Data fitting, model training, and prediction plotting.
- Output: Comparative plots of predicted vs actual temperatures.

---

## 📌 Applications

- Battery Management Systems (BMS)
- Predictive Maintenance
- Electric Vehicle (EV) Thermal Monitoring

---

## 👤 Author

**Harshit Chaudhary**

🔗 [GitHub Profile](https://github.com/Harshit-Chaudhry)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
