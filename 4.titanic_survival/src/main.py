import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv("4.titanic_survival/data/train.csv")
test_data = pd.read_csv("4.titanic_survival/data/test.csv").drop("Name", axis=1)

combined_data = pd.concat([train_data.drop("Survived", axis=1), test_data], axis=0)

# Process data, converting to floats and dropping columns
combined_data["isMale"] = combined_data["Sex"].map({"male": 1, "female": 0})
combined_data.drop("Sex", axis=1, inplace=True)
combined_data = pd.get_dummies(combined_data, columns=["Embarked"], prefix="Embarked")
combined_data["HasCabin"] = combined_data["Cabin"].apply(lambda x: 0 if pd.isna(x) else 1)
combined_data.drop("Cabin", axis=1, inplace=True)
combined_data["Age"].fillna(combined_data["Age"].median(), inplace=True)
combined_data["Fare"].fillna(combined_data["Fare"].median(), inplace=True)
combined_data.drop("Ticket", axis=1, inplace=True)
combined_data.drop("PassengerId", axis=1, inplace=True)
combined_data.drop("Name", axis=1, inplace=True)

X_train = combined_data.iloc[:len(train_data)]
X_test = combined_data.iloc[len(train_data):]

y_train = train_data["Survived"]
y_test = pd.read_csv("4.titanic_survival/data/test_result.csv").drop("PassengerId", axis=1)

dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)

print("Decision Tree:")
print(accuracy_score(y_test, y_pred_dt))