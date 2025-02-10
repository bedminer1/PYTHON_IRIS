import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("data/iris.csv")

le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])
X = data.drop('species', axis=1)
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Decision Tree
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)

print("Decision Tree:")
print(accuracy_score(y_test, y_pred_dt))

# Random Forest
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

print("Random Forest:")
print(accuracy_score(y_test, y_pred_rf))

# SVM
sv_classifier = SVC()
sv_classifier.fit(X_train, y_train)
y_pred_sv = sv_classifier.predict(X_test)

print("State Vector Machine:")
print(accuracy_score(y_test, y_pred_sv))

# Plotting
sns.pairplot(data, hue="species")
plt.savefig("plots/pairplot.png")
plt.show()
