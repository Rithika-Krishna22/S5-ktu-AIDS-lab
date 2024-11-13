from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=43)
clf=KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("accuracy:" ,accuracy_score(y_test,y_pred))
print("classification report:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print('\nCORRECT PREDICTIONS:')
for i in range(len(y_pred)):
	if y_pred[i]==y_test[i]:
		print('actual:',y_test[i],'predicted:',y_pred[i])
print('\nWRONG PREDICTIONS:')
for i in range(len(y_pred)):
	if y_pred[i]!=y_test[i]:
		print('actual:',y_test[i],'predicted:',y_pred[i])