from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
data=load_iris()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
clf=SVC(kernel='linear')
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print('Accuracy:',accuracy_score(y_test,y_pred))
print('Classification report:', classification_report(y_test,y_pred))
conf=confusion_matrix(y_test,y_pred)
sns.heatmap(conf,annot=True,cmap='Blues',xticklabels=data.target_names,yticklabels=data.target_names)
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
plt.show()
