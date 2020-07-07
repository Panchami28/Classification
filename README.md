## KNN classification algorithm

### Introduction
**K-Nearest Neighbors (KNN)** is one of the simplest algorithms used in Machine Learning for regression and classification problem. KNN algorithms use data and classify new data points based on similarity measures (e.g. distance function).Here classification is done by a majority vote to its neighbors.K-NN is a non-parametric algorithm, which means it does not make any assumption on underlying data. It is also called a lazy learner algorithm because it does not learn from the training set immediately instead it stores the dataset and at the time of classification, it performs an action on the dataset. KNN algorithm at the training phase just stores the dataset and when it gets new data, then it classifies that data into a category that is much similar to the new data.

**Example:** Suppose, we have an image of a creature that looks similar to cat and dog, but we want to know either it is a cat or dog. So for this identification, we can use the KNN algorithm, as it works on a similarity measure. Our KNN model will find the similar features of the new data set to the cats and dogs images and based on the most similar features it will put it in either cat or dog category.

### Implementation of KNN using python

Importing the standard libraries

```markdown
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```

Loading the dataset

```markdown
cars=pd.read_csv("mtcars.txt")
cars.head()
cars.columns

#checking for missing values
cars.isnull().sum()
```

Processing the data

```markdown
X=cars.loc[:,['mpg','wt','hp','gear','cyl']]
y=cars.loc[:,'am']

from sklearn import preprocessing 
X=preprocessing.scale(X)
```
Splitting the dataset into training and testing

```markdown
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)
X_train.shape
```

Trainig the model using KNN

```markdown
from sklearn.neighbors import KNeighborsClassifier
model_knn =KNeighborsClassifier(n_neighbors=1)
#Training the model
model_knn.fit(X_train,y_train)
```

Prediction using test data

```markdown
y_predict = model_knn.predict(X_test)
```

checking the model accuracy

```markdown
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_test,y_predict)
print(classification_report(y_test,y_predict))
```
A quick comparison with different values of k

```markdown
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print('WITH K=5')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
```

WITH K=1
`
              precision    recall  f1-score   support

           0       1.00      0.86      0.92         7
           1       0.75      1.00      0.86         3

    accuracy                           0.90        10
   macro avg       0.88      0.93      0.89        10
weighted avg       0.93      0.90      0.90        10
`

WITH K=5
`
      precision    recall  f1-score   support

           0       0.71      0.71      0.71         7
           1       0.33      0.33      0.33         3

    accuracy                           0.60        10
   macro avg       0.52      0.52      0.52        10
weighted avg       0.60      0.60      0.60        10
`

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Panchami28/class/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
