1\. Project Overview



This project builds a credit default prediction model using the UCI Default of Credit Card Clients dataset. The primary goal is to detect clients with high default risk using machine learning, and evaluate the model using metrics such as precision, recall, ROC AUC, confusion matrix, SHAP explainability, and threshold sensitivity analysis.



The model chosen is Random Forest Classifier, with additional threshold tuning to improve recall for defaulters.



====================================================



2\. Environment \& Dependencies



Required Packages:



Install the following dependencies before running the code: 

**pip install pandas numpy matplotlib seaborn scikit-learn shap**



====================================================



3\. File Structure



Program/

│── train\_rf.ipynb             # Jupyter Notebook (must keep execution results)

│── train\_rf.py                # Main Python script

│── readme.md                  # This file

│

├── data/

│     └── uci\_default\_cleaned.csv

│

└── result/

&nbsp;     ├── confusion\_matrix\_final.png

&nbsp;     ├── roc\_curve\_final.png

&nbsp;     ├── shap\_importance\_bar.png

&nbsp;     ├── shap\_summary\_plot.png

&nbsp;     ├── threshold\_comparison.png

&nbsp;     └── threshold\_sensitivity\_analysis.csv



====================================================



4\. How to Run the Code



Option A: Execute under Program/: **python train\_rf.py**



This program will automatically perform the following steps:



&nbsp;	1.Read data



&nbsp;	2.Splitting the training/test set



&nbsp;	3.Training Random Forest



&nbsp;	4.Adjusting the threshold (finding Recall ≈ 0.6)



&nbsp;	5.Generating a model evaluation report



&nbsp;	6.Outputting all graphs to /result/



&nbsp;	7.Saving the threshold sensitivity analysis table



Option B: Run Jupyter Notebook: **train\_rf.pynb**



The Notebook version includes complete output for easy workflow review.



====================================================



5\. Output Files



| File                               | Description                             |

| ---------------------------------- | --------------------------------------- |

| confusion\_matrix\_final.png         | Confusion matrix (threshold-adjusted)   |

| roc\_curve\_final.png                | ROC curve \& AUC                         |

| shap\_importance\_bar.png            | SHAP feature importance                 |

| shap\_summary\_plot.png              | SHAP summary plot                       |

| threshold\_comparison.png           | Precision/Recall/Accuracy vs. threshold |

| threshold\_sensitivity\_analysis.csv | Threshold performance table             |































