# Week 11: Logistic Regression Car Purchase Assignment

This folder contains the completed notebook for the INFO 4670 Module 7 regression assignment.

## Run

Download `car_purchase_data.csv` from the course assignment page and place it in the same working directory as the notebook. Then open `Logistic_Regression_Car_Assignment.ipynb` in Jupyter or Google Colab. The notebook uses pandas, NumPy, Matplotlib, and scikit-learn. It uses `Age` and `Salary` to predict `Purchased`; the identifier and `Gender` fields are not included in this two-feature model.

The notebook's fixed 80/20 holdout split achieved **92.5% accuracy (74/80)**. Results are reported along with the confusion matrix, classification report, and decision-region visualization.