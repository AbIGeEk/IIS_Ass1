import sklearn
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import manifold, neighbors, metrics
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as pyplot
import matplotlib.cm as cm
import matplotlib.pyplot
SIMPLE_EMBEDDING=False
MANUAL_SPLIT=True
DATA_2D=True

def plotdata(X):

    if(DATA_2D):

        pyplot.scatter(X[:, 0],X[:,1],c=color_array)
        pyplot.show()

    else: 
        fig = pyplot.figure()
        ax = Axes3D(fig)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        pyplot.show()


filename = "/media/Local-Disk/Abishek/Labs/Inteligent\ Interactive\ Systems/Assignment\ 1/dataset/Face-Feat-with-image-file-names.txt"
face_features = np.loadtxt(filename, delimiter=",", usecols=range(2,24))
face_labels = np.genfromtxt(filename, dtype=None, delimiter=",\t", usecols=range(1,2))
filename2 = "/media/Local-Disk/Abishek/Labs/Intelligent\ Interactive\ Systems/Assignment\ 1/dataset/HogFeat-with-image-file-names-txt"
hog_features = np.loadtxt(filename2, delimiter=",", usecols=range(2,102))

color_array = np.array([])
print(color_array.shape)
for label in face_labels: 
    if(label == "POSITIVE"):
        color_array = np.append(color_array, "red")
    if(label == "NEGETIVE"):
        color_array = np.append(color_array, "greeen")
    if(label == "NEUTRAL"):
        color_array = np.append(color_array, "blue")
print(color_array.shape)
combined_features = np.append(face_features, hog_features, axis=1)
data = combined_features.reshape(68, -1)
pca = PCA(n_components=2)
x_trans = pca.fit_transform(data)
if(DATA_2D):
    nComp = 2
else:
    nComp = 3

if(SIMPLE_EMBEDDING):
    pca = PCA(n_components = 2)
    x_trans = pca.fit_transform(data)
    print(x_trans)
else:
    tsne = manifold.TSNE(n_components = 2, init='pca', random_state=0)
    x_trans = tsne.fittransform(data)

# Plotting the data
plotData(X_trans)

X_train, X_test, Y_train, Y_test = train_test_split(X_trans, face_labels, test_size=0.33, random_states=5)

n_neighbors = 5
# kNN Classifier
kNNClassifier = neighbors.KNeighborsClassifier(n_neighbors, weights="distance")

#Training the model
kNNClassifier.fit(X_trans, Y_train)
predictedLabels = kNNClassifier.predict(X_test)
print("Classification report for classifier %s:\n%s\n" % ('k-NearestNeighbour', metrics.classification_report(Y_test, predictedLabels)))
print("COnfusion matrix:\n%s" % metrics.confusion_matrix(Y_test, predictedLables))
scores = cross_val_score(kNNClassifier, combined_features, face_labels, cv=5)
print(scores)

#SVM Classifier
clf = SVC(gamma='auto')
clf.fit(X_train, Y_train)
SVC(c=0.4, cahce_size=500, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
predictedSVMLables = clf.predict(X_test)

acc_svm = metrics.accuracy_score(Y_test, predictedSVMLables)
print ("Linear SVM Accuracy: "),acc_svm

# Cross Validation
scores = cross_val_score(clf, combined_features, face_labels, cv=5)
print(scores)
