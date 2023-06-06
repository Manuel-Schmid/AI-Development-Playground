from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load breast cancer dataset
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
clf = RandomForestClassifier(random_state=25)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

#  print plot with confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='g')
plt.ylabel('Prediction', fontsize=14)
plt.xlabel('Actual', fontsize=14)
plt.title('Confusion Matrix', fontsize=17)
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy   :", accuracy)
