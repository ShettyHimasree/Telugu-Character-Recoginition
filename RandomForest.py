import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pickle
#getting the file
df = pd.read_csv(r'Processedtrain500_12.csv', encoding='UTF-8')
dft = pd.read_csv(r'Processedtest_12.csv')
df['36']=df['36'].replace(['a','aa','e','ee','u'],[1,2,3,4,5])
# print(df)

train_data=df
test_data=dft

# print(f"No. of training examples: {train_data.shape[0]}")
# print(f"No. of testing examples: {test_data.shape[0]}")
X_train_data, y_train_data,X_test_data,y_test_data = train_data.iloc[:,:-1].values,train_data.iloc[:,-1].values,test_data.iloc[:,:-1],test_data.iloc[:,-1]

#stratified k fold

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
test_accs = []
best_train_data = None
best_test_data = None
for fold, (train_idx, test_idx) in enumerate(skf.split(X_train_data, y_train_data)):
    print(f'Fold {fold+1}')
    X_train, y_train = X_train_data[train_idx], y_train_data[train_idx]
    X_test, y_test = X_train_data[test_idx], y_train_data[test_idx]
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_accs.append(test_acc)
    
    print(f'Validation accuracy: {test_acc:.3f}')
    
    if best_test_data is None or test_acc > accuracy_score(best_test_data[1], clf.predict(best_test_data[0])):
        best_trainie = (X_train, y_train)
        best_testie = (X_test, y_test)
avg_test_acc = sum(test_accs) / len(test_accs)

x_cross = best_trainie[0]
y_cross = best_trainie[1]
x_test_cross = best_testie[0]
y_test_cross = best_testie[1]
clf = RandomForestClassifier()
# print(clf)
# print('----------------------------------------------------')
clf.fit(x_cross, y_cross) 
# y_pred = clf.predict(x_test_cross)

# print("Accuracy:", accuracy_score(y_test_cross, y_pred))

#make pickle file
pickle.dump(clf,open("RandomForest.pkl","wb"))