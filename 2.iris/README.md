#### Mission
Given labeled data (features and corresponding species), create a model that can guess the species of an unidentified flower given its features. 

#### Implementation
- **Label Encoding**: The species label is converted into numerical representations using LabelEncoder from scikit-learn. Necessary as machine learning algorithms typically work with numerical data.
- `train_test_split` splits into training and testing sets. `stratify` parameter is used to maintain the class balance in both training and testing sets. 
- **Feature Scaling**: `StandardScaler` is used to standardize the features (mean=0, standard deviation=1). This is crucial for SVMs because they are sensitive to scales, less necessary for decision trees and random forest but still good practice.
- **Model Training and Evaluation:** models were instantiated then `fit()` was called with training data to train the models, `predict()` for making predictions on test dataset, and `accuracy_score()` was called to compared predicted results to real results and evaluate the model. 
 
#### Models
- **Decision Tree:** A tree-based model that makes decisions based on feature values. Easy to interpret but prone to overfitting.
- **Random Forest:** An ensemble method that combines multiple decision trees. More robust and less prone to overfitting than single decision trees. Can provide feature importance estimates.
- **Support Vector Machine (SVM):** Finds the optimal hyperplane to separate classes. Effective in high-dimensional spaces and can handle non-linearly separable data using kernels. Can be less interpretable and computationally intensive.