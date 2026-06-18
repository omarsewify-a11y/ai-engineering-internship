# Week 3 Day 1 Report – Iris Dataset

## Data Preparation
In this task, the Iris dataset was loaded directly from scikit-learn. The dataset is already clean and structured, so no missing value handling or encoding was required. The data was converted into a DataFrame for easier processing.

The dataset was split into features (flower measurements) and the target variable (flower species). An 80/20 train-test split was applied to ensure that the model was evaluated on unseen data.

## Model Training
A Decision Tree Classifier was used to learn patterns from the training data. The model was then tested on the unseen test data to evaluate its performance.

## Evaluation Results
The model was evaluated using:
- Accuracy Score
- Confusion Matrix
- Classification Report

The accuracy score shows the overall percentage of correct predictions. The confusion matrix shows how predictions are distributed across correct and incorrect classes. The classification report provides precision, recall, and F1-score for each class.

## Interpretation
The confusion matrix shows how well the model distinguishes between the three iris species. Most predictions fall on the diagonal, meaning they are correct. A few misclassifications are present between similar classes.

Precision shows how often predicted classes are correct. Recall shows how many actual samples were correctly identified. F1-score balances both precision and recall.

Overall, the model performs well because the Iris dataset has clear patterns that are easy for decision trees to learn. This makes it a good dataset for understanding evaluation metrics.
