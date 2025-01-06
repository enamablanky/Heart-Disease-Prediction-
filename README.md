# **Heart Disease Prediction Using Machine Learning**  

## **Overview**  
This project focuses on predicting the likelihood of **heart disease** based on various clinical and demographic features. By leveraging machine learning algorithms, the model aims to assist healthcare professionals in identifying at-risk patients early, enabling timely intervention and treatment.  

---

## **Problem Statement**  
Heart disease remains one of the leading causes of death worldwide. Accurate prediction models can significantly aid in **early diagnosis** and **preventive healthcare**.  

**Goal:** Develop a machine learning model that can predict whether a person has heart disease based on input features such as **age, cholesterol levels, blood pressure**, and other cardiovascular indicators.  

---

## **Dataset**  

**Source:** Publicly available heart disease dataset (e.g., UCI Machine Learning Repository).  

**Features:**  
1. **Age** – Patient’s age in years.  
2. **Sex** – Gender (1 = Male, 0 = Female).  
3. **Chest Pain Type (CP)** – Types of chest pain experienced (categorical).  
4. **Resting Blood Pressure (trestbps)** – Patient’s resting blood pressure (in mm Hg).  
5. **Cholesterol (chol)** – Serum cholesterol level (in mg/dl).  
6. **Fasting Blood Sugar (fbs)** – Whether fasting blood sugar is > 120 mg/dl (1 = True, 0 = False).  
7. **Resting ECG (restecg)** – Results of resting electrocardiogram (categorical).  
8. **Max Heart Rate (thalach)** – Maximum heart rate achieved.  
9. **Exercise-Induced Angina (exang)** – Whether angina was induced during exercise (1 = Yes, 0 = No).  
10. **Oldpeak** – ST depression induced by exercise relative to rest.  
11. **Slope** – Slope of the peak exercise ST segment.  
12. **Thalassemia (thal)** – Blood disorder type (categorical).  

**Target Variable:**  
- **Presence of Heart Disease (target)** – Binary outcome (1 = Disease, 0 = No Disease).  

---

## **Methodology**  

### **Step 1: Data Preprocessing**  
- **Missing Values Handling:** Checked for and imputed missing values.  
- **Categorical Encoding:** Converted categorical variables using **One-Hot Encoding**.  
- **Feature Scaling:** Normalized numerical data to standardize features.  
- **Outlier Detection:** Identified and treated outliers using boxplots and Z-scores.  

### **Step 2: Exploratory Data Analysis (EDA)**  
- **Visualizations:**  
  - **Histograms and Bar Plots** to analyze feature distributions.  
  - **Correlation Heatmaps** to identify relationships between features and target.  
  - **Boxplots** for outlier detection.  
- **Feature Relationships:** Studied relationships between features like age, cholesterol, and heart disease incidence.  

### **Step 3: Model Development**  
- **Machine Learning Models Tested:**  
  1. **Logistic Regression** – Baseline classifier.  
  2. **K-Nearest Neighbors (KNN)** – Non-linear classifier.  
  3. **Support Vector Machine (SVM)** – Effective for high-dimensional spaces.  
  4. **Random Forest** – Ensemble method for improving accuracy.  
  5. **Gradient Boosting (XGBoost)** – Optimized model for better performance.  

- **Model Selection:**  
  - Compared models using cross-validation.  
  - Fine-tuned hyperparameters using **GridSearchCV** and **RandomizedSearchCV**.  

### **Step 4: Model Evaluation**  
- **Performance Metrics:**  
  - **Accuracy:** Measures overall correctness.  
  - **Precision and Recall:** Evaluates positive prediction correctness and completeness.  
  - **F1-Score:** Balances precision and recall.  
  - **ROC-AUC Score:** Evaluates classifier performance based on sensitivity and specificity.  

---

## **Results and Insights**  

- Achieved **accuracy > 85%** using ensemble methods like **Random Forest** and **Gradient Boosting**.  
- Identified **age, chest pain type, and cholesterol levels** as the most influential factors for predicting heart disease.  
- Highlighted the importance of **exercise-induced angina** and **ST depression** in heart disease diagnosis.  
- Created a **scalable and reusable framework** for healthcare predictions.  

---

## **Technologies Used**  

- **Programming Language:** Python  
- **Libraries and Tools:**  
  - **Data Handling:** Pandas, NumPy  
  - **Visualization:** Matplotlib, Seaborn  
  - **Modeling:** Scikit-learn, XGBoost  
  - **Environment:** Jupyter Notebook  

---

## **Applications**  

- **Healthcare Industry:** Early identification of at-risk patients for preventive care.  
- **Insurance Sector:** Risk assessment for insurance policy pricing.  
- **Clinical Decision Support Systems:** Assistance to doctors for faster diagnosis.  
- **Public Health:** Analysis of health trends for policymaking.  

---

## **Challenges and Solutions**  

1. **Imbalanced Data:**  
   - Applied **resampling techniques** to balance positive and negative classes.  

2. **Feature Selection:**  
   - Used **correlation matrices** and **feature importance scores** to reduce irrelevant features.  

3. **Overfitting:**  
   - Prevented overfitting by employing **cross-validation** and **regularization techniques** in models.  

---

## **Future Enhancements**  

1. **Deep Learning Models:** Incorporate neural networks (e.g., TensorFlow/Keras) for better performance.  
2. **Deployment:** Create a **web-based interface** using Flask or Streamlit for real-time predictions.  
3. **Feature Expansion:** Include additional data, such as dietary habits and lifestyle factors, for improved predictions.  
4. **Explainability Tools:** Use **SHAP values** and **LIME** for model interpretability.  

---

## **License**  

This project is licensed under the **MIT License**, allowing free usage and modification with proper attribution.  

---

## **Contributors**  

- **[Enam Nsopulu Obiosio]** – Data Analysis, Model Development, Documentation.  
- Open to contributions. Feel free to submit pull requests!  

---

## **Conclusion**  

This project demonstrates the application of machine learning in **healthcare analytics**. By leveraging data preprocessing, visualization, and advanced models, it predicts heart disease risk with high accuracy. The insights gained can support **early diagnosis** and **preventive healthcare planning**, showcasing the impact of AI in medicine. 
