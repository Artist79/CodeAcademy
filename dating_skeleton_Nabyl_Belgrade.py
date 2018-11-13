### Author: Nabyl Belgrade
### Project: CodeAcademy Capstone: Date-A-Scientist
### Version: 1.0
### Date: 13-Nov-2018

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

import sklearn.preprocessing as preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score

import os


#Import your all_data here
root_path = os.path.join(os.path.expanduser("~"), "CA_Capstone")
filePath = os.path.join(root_path, "profiles.csv")

### Load in the DataFrame
path_file = "Macintosh HD\\Users\\Nabs\\CA_Capstone\\"
all_data = pd.read_csv(filePath)

### Explore the Data
print "age"
print(all_data.age.head())
print
print all_data.age.mean()
print all_data.age.median()
print all_data.age.var()
print
print(all_data.age.value_counts())
print

print "income"
print(all_data.income.head())
print
print all_data.income.mean()
print all_data.income.median()
print all_data.income.var()
print
print(all_data.income.value_counts())
print

print "body_type"
print(all_data.speaks.value_counts())
print

print "speaks"
print(all_data.speaks.value_counts())
print

### Visualize some of the Data
plt.hist(all_data.age, bins=20, alpha=.5, color='green')
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

all_data = all_data[all_data.income<=500000]
plt.hist(all_data.income, alpha=.5, color='red')
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()

all_data['body_type'].value_counts().plot(kind='barh', alpha=.5)
plt.show()

all_data['education'].value_counts().plot(kind='barh', alpha=.5, fontsize=6)
plt.show()

all_data['drugs'].value_counts().plot(kind='barh', alpha=.5)
plt.show()
### Formulate a Question

### Augment your Data
all_languages = [elt.split(',') if type(elt) == str else [] for elt in all_data["speaks"].tolist() ]
for languages in all_languages:
    languages = [language for language in languages if not "(poorly)" in language]
    languages = [language for language in languages if not "C++" in language]

all_data["nb_languages"] = [len(languages) for languages in all_languages]

print "nb_languages"
print all_data.nb_languages.mean()
print all_data.nb_languages.median()
print all_data.nb_languages.var()
print
print(all_data.nb_languages.value_counts())
print

all_data['nb_languages'].value_counts().plot(kind='barh', alpha=.5)
plt.xlabel("number of languages")
plt.ylabel("Frequency")
plt.show()

essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
all_essays = all_data[essay_cols].replace(np.nan, '', regex=True)
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
all_data["essay_len"] = all_essays.apply(lambda x: len(x))

print "essay_len"
print all_data.essay_len.mean()
print all_data.essay_len.median()
print all_data.essay_len.var()
print
print(all_data.essay_len.value_counts())
print

plt.hist(all_data.essay_len, alpha=.5, color='green')
plt.xlabel("essay_len")
plt.ylabel("Frequency")
plt.show()

### Clean data
all_data = all_data.dropna(subset=["income", "age", "body_type", "speaks", "education", "drugs"])

all_data.income = all_data.income.astype(float)
all_data = all_data[all_data.income>0]
all_data = all_data[all_data.income<=800000]

all_data.age = all_data.age.astype(float)
all_data = all_data[all_data.age<=80]

all_data = all_data[all_data.body_type!="rather not say"]

all_data = all_data[all_data.nb_languages>0]

all_data = all_data[all_data.essay_len<=20000]

all_data = all_data[all_data.education!="rather not say"]
all_data = all_data[~all_data['education'].str.contains('|'.join(["working"]))]

### Data mapping
body_type_mapping = {"skinny": 0, "thin": 1,"average": 2, "fit": 3, "athletic": 3, "curvy": 4, "a little extra": 5, "full figured": 6, "overweight": 7, "jacked": 8, "used up": 9, "rather not say": -1}
all_data["body_type_code"] = all_data.body_type.map(body_type_mapping)

drug_mapping = {"never": 0, "sometimes": 1, "often": 2}
all_data["drugs_code"] = all_data.drugs.map(drug_mapping)

education_mapping = {
"graduated from ph.d program":          0,
"graduated from space camp":            0,
"graduated from law school":            0,
"graduated from med school":            0,

"ph.d program":                         0,
"space camp":                           0,
"law school":                           0,
"med school":                           0,
"masters program":                      0,

"working on ph.d program":              1,
"working on space camp":                1,
"working on law school":                1,
"working on med school":                1,
"working on masters program":           1,

"dropped out of ph.d program":          2,
"dropped out of space camp":            2,
"dropped out of law school":            2,
"dropped out of med school":            2,

"graduated from college/university":    2,
"graduated from masters program":       2,

"college/university":                   2,
"working on college/university":        2,

"dropped out of college/university":    3,
"dropped out of masters program":       3,

"graduated from two-year college":      4,
"two-year college":                     4,
"working on two-year college":          4,
"dropped out of two-year college":      5,

"graduated from high school":           5,
"high school":                          5,
"working on high school":               5,
"dropped out of high school":           6,
}
all_data["education_code"] = all_data.education.map(education_mapping)

### Normalize Data
feature_data = all_data[['age', 'income', 'body_type_code', 'essay_len', 'nb_languages', 'drugs_code', 'education_code']]
min_max_scaler = preprocessing.MinMaxScaler()
feature_data_scaled = min_max_scaler.fit_transform(feature_data.values)
normalized_data = pd.DataFrame(feature_data_scaled, columns=feature_data.columns)

X = ["age", "income", "body_type_code", "essay_len", "nb_languages", "drugs_code"]
Y =["education_code"]

for x in X:
    plt.hist(normalized_data[x], alpha=.3, color='purple')
    plt.xlabel(x)
    plt.ylabel("Frequency")
    plt.show()

plt.hist(feature_data[Y[0]].tolist(), alpha=.4, color='orange')
plt.xlabel(Y[0])
plt.ylabel("Frequency")
plt.show()

for x in X:
    for y in Y:
        plt.scatter(feature_data[x], feature_data[y], alpha=0.4, c='grey', cmap=cm.jet)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

### Use Classification Techniques
training_data, validation_data, training_labels, validation_labels = train_test_split(normalized_data[X], feature_data[Y[0]], test_size=0.2, random_state=10)
feature_data = feature_data.reset_index(drop=True)

case = 2

scores = []
f1_scores = []
precision_scores = []

if case==1:

    for k in range(1, 1000, 9):
        k_classifier = KNeighborsClassifier(n_neighbors=k, weights = "distance")
        k_classifier.fit(training_data, training_labels)
        predicted_labels = k_classifier.predict(validation_data)

        scores.append(k_classifier.score(validation_data, validation_labels))
        f1_scores.append(f1_score(validation_labels.tolist(), predicted_labels, average='weighted'))
        precision_scores.append(precision_score(validation_labels.tolist(), predicted_labels, average='weighted'))

        print k, "\t", scores[-1], "\t", f1_scores[-1], "\t", precision_scores[-1]

    k_list = [k for k in range(1, 1000, 9)]

    plt.plot(k_list, scores, color='red')
    plt.xlabel("k")
    plt.ylabel("Validation Accuracy")
    plt.show()

    plt.plot(k_list, precision_scores, color='blue')
    plt.xlabel("k")
    plt.ylabel("precision_score")
    plt.show()

    plt.plot(k_list, f1_scores, color='green')
    plt.xlabel("k")
    plt.ylabel("f1_score")
    plt.show()

else:
    for C in range(5, 1000, 50):
        svm_classifier  = SVC(kernel = 'rbf', probability = True, gamma = 'auto', C = C/100.)
        svm_classifier.fit(training_data, training_labels)
        predicted_labels = svm_classifier.predict(validation_data)

        scores.append(svm_classifier.score(validation_data, validation_labels))
        f1_scores.append(f1_score(validation_labels.tolist(), predicted_labels, average='weighted'))
        precision_scores.append(precision_score(validation_labels.tolist(), predicted_labels, average='weighted'))

        print C/100., "\t",  scores[-1], "\t", f1_scores[-1], "\t", precision_scores[-1]

    C_list = [C/100. for C in range(5, 1000, 50)]

    plt.plot(C_list, scores, color='red')
    plt.xlabel("C" )
    plt.ylabel("Validation Accuracy")
    plt.show()

    plt.plot(C_list, precision_scores, color='blue')
    plt.xlabel("k")
    plt.ylabel("precision_score")
    plt.show()

    plt.plot(C_list, f1_scores, color='green')
    plt.xlabel("k")
    plt.ylabel("f1_score")
    plt.show()
