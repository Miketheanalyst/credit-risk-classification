# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
  - The purpose of the analysis was to build a predictive model using logistic regression to identify high-risk loans from a dataset of historical lending activity. The goal was to determine whether a loan is healthy or at high risk of default.

* Explain what financial information the data was on, and what you needed to predict.
  - The data included financial information such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, and total debt. The target variable to predict was `loan_status`, indicating whether a loan is healthy (0) or high-risk (1).

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
  - The `loan_status` variable had 75,036 healthy loans and 2,500 high-risk loans, showing a significant class imbalance.

* Describe the stages of the machine learning process you went through as part of this analysis.
  - **Data Loading and Preparation:** Loaded the dataset, separated the features and target variable, and split the data into training and testing sets.
  - **Model Training:** Trained a logistic regression model using the training data.
  - **Model Prediction:** Used the trained model to make predictions on the testing data.
  - **Model Evaluation:** Evaluated the model's performance using a confusion matrix and classification report.

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any other algorithms).
  - Used the `LogisticRegression` algorithm from scikit-learn to train the model. Utilized the `train_test_split` function to split the data and the `confusion_matrix` and `classification_report` functions to evaluate the model.

## Results

Using bulleted lists, describe the accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores:
    - **Confusion Matrix:**

    *Interpretation:*
    
      ```
      [[15040     1]
       [   46    25]]


      True Negatives (15040):

The model correctly identified 15,040 healthy loans as healthy. This is a good outcome because these are loans that are not at risk, and the model recognized them as such.
False Positives (1):

The model incorrectly identified 1 healthy loan as high-risk. This means one borrower might be incorrectly flagged as high-risk, which could lead to unnecessary precautions or actions being taken against a healthy borrower.
False Negatives (46):

The model incorrectly identified 46 high-risk loans as healthy. This is a more concerning outcome because these high-risk loans were not identified as such, which means potential financial risk was not mitigated. These are missed opportunities for intervention.
True Positives (25):

The model correctly identified 25 high-risk loans as high-risk. This is a positive outcome because these high-risk loans can be managed or mitigated accordingly, reducing financial risk.
    
    
    - **Classification Report:**
      ```
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00     15041
               1       0.96      0.35      0.51        71

        accuracy                           1.00     15112
       macro avg       0.98      0.68      0.75     15112
    weighted avg       1.00      1.00      1.00     15112
      ``
      
     *Interpretation:*


Class 0 (Healthy Loans):
Precision: 1.00

Precision is the ratio of correctly predicted positive observations to the total predicted positives. A precision of 1.00 for class 0 means that all loans predicted as healthy by the model are actually healthy.
\text{Precision}_0 = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} = \frac{15040}{15040 + 1} \approx 1.00
]

Recall: 1.00

Recall is the ratio of correctly predicted positive observations to the all observations in actual class. A recall of 1.00 for class 0 means that the model successfully identifies all actual healthy loans.
\text{Recall}_0 = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} = \frac{15040}{15040 + 0} \approx 1.00
]

F1-Score: 1.00

The F1-score is the weighted average of precision and recall. Since both precision and recall are 1.00, the F1-score is also 1.00, indicating perfect performance for identifying healthy loans.
\text{F1-Score}_0 = 2 \times \frac{\text{Precision}_0 \times \text{Recall}_0}{\text{Precision}_0 + \text{Recall}_0} = 2 \times \frac{1.00 \times 1.00}{1.00 + 1.00} = 1.00
]

Support: 15,041

Support is the number of actual occurrences of the class in the dataset. There are 15,041 healthy loans in the test set.
Class 1 (High-Risk Loans):
Precision: 0.96

A precision of 0.96 for class 1 means that 96% of the loans predicted as high-risk by the model are actually high-risk. This is quite high, indicating that when the model predicts a loan as high-risk, it is usually correct.
\text{Precision}_1 = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} = \frac{25}{25 + 1} \approx 0.96
]

Recall: 0.35

A recall of 0.35 for class 1 means that the model correctly identifies 35% of the actual high-risk loans. This indicates that the model misses a significant number of high-risk loans.
\text{Recall}_1 = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} = \frac{25}{25 + 46} \approx 0.35
]

F1-Score: 0.51

The F1-score for class 1 is 0.51, which is a balance between precision and recall. This lower score reflects the trade-off between having high precision and low recall.
\text{F1-Score}_1 = 2 \times \frac{\text{Precision}_1 \times \text{Recall}_1}{\text{Precision}_1 + \text{Recall}_1} = 2 \times \frac{0.96 \times 0.35}{0.96 + 0.35} \approx 0.51
]

Support: 71

There are 71 high-risk loans in the test set.
Overall Metrics:
Accuracy: 1.00

Accuracy is the overall ratio of correctly predicted observations to the total observations. An accuracy of 1.00 means the model correctly predicted the loan status for 100% of the test cases.
\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Observations}} = \frac{15040 + 25}{15112} \approx 1.00
]

Macro Average:

Precision: 0.98
This is the average precision of the two classes.
Recall: 0.68
This is the average recall of the two classes, showing that the model's ability to identify both classes correctly is not balanced.
F1-Score: 0.75
This is the average F1-score of the two classes.
Weighted Average:

Precision: 1.00
This takes into account the number of instances of each class to calculate a weighted precision.
Recall: 1.00
This takes into account the number of instances of each class to calculate a weighted recall.
F1-Score: 1.00
This takes into account the number of instances of each class to calculate a weighted F1-score.
## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

* Which one seems to perform best? How do you know it performs best?
  - The logistic regression model performs well in predicting healthy loans, with precision, recall, and F1-score all being 1.00. However, it performs poorly in predicting high-risk loans, with a recall of only 0.35 and an F1-score of 0.51. This indicates that while the model is highly accurate overall, it is less reliable in identifying high-risk loans due to the class imbalance.

* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s?)
  - Performance depends on the problem we are trying to solve. In this case, identifying high-risk loans (class 1) is crucial, as it helps mitigate financial risks. Therefore, improving recall for high-risk loans is important, even if it slightly reduces the accuracy for healthy loans (class 0).

If you do not recommend any of the models, please justify your reasoning.
- While the logistic regression model is not perfect, it provides a strong starting point. However, due to its poor recall for high-risk loans, it is recommended to further explore methods to handle class imbalance, such as resampling techniques, adjusting class weights, or using more sophisticated models like ensemble methods to improve the detection of high-risk loans.

---

Feel free to adjust any parts of this template as per your specific needs or additional findings. If you have any more questions or need further assistance, let me know!

