from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import local_binary_pattern
from skimage import color
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, balanced_accuracy_score
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pickle

def grayscaling(image):
    image = image[:,:,:3]
    gray_img = color.rgb2gray(image)
    return gray_img

def lbp(image):
    gray = grayscaling(image)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    return lbp.ravel()

def load_image(container_path, dimension=(64,128)):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    lbp_feature = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            lbp_feature.append(lbp(img_resized))
            target.append(i)
    target = np.array(target)

    return Bunch(data=lbp_feature,
                 target=target,
                 target_names=categories)

def pca(data, target_labels):
    pca = PCA(n_components=0.95)
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    initial_dimension = scaler.n_features_in_
    reduced_data = pca.fit_transform(data_normalized)

    num_dimensions_retained = pca.n_components_
    print("Number of initial dimensions:", initial_dimension)
    print("Number of retained dimensions:", num_dimensions_retained)

    explained_variance_ratios = pca.explained_variance_ratio_
    print("Explained Variance Ratios:", explained_variance_ratios)

    return reduced_data

def make_model(filename, X_train, y_train):
    param_grid = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'kernel': ['rbf','linear','poly']}]
    scoring_metric = make_scorer(balanced_accuracy_score)
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid, scoring=scoring_metric)
    clf.fit(X_train, y_train)

    with open(filename, 'wb') as file:
        pickle.dump(clf, file)

    y_pred = clf.predict(X_train)
    print("Training result:\n")
    print("Parameters used : ", clf.best_params_)
    print("\nClassification report :\n", classification_report(y_train, y_pred))
    print("\nConfusion matrix: \n", confusion_matrix(y_train, y_pred))

def testing(model_name, X_test, y_test):
    with open(model_name, 'rb') as file:
        loaded_svm_model = pickle.load(file)

    y_pred = loaded_svm_model.predict(X_test)

    print("Testing result:\n")
    print("Classification report :\n", classification_report(y_test, y_pred))
    print("\nConfusion matrix: \n", confusion_matrix(y_test, y_pred))

dataset = load_image('dataset_path')

X_train, X_test, y_train, y_test = train_test_split(dataset.data, 
                                                    dataset.target, 
                                                    test_size=0.3,
                                                    random_state=109)

make_model('model_name.pkl', X_train, y_train)
testing('model_name.pkl', X_test, y_test)

dataset.data = pca(dataset.data, dataset.target)
X_train2, X_test2, y_train2, y_test2 = train_test_split(dataset.data,
                                                    dataset.target,
                                                    test_size=0.3,
                                                    random_state=109)

make_model('pca_model_name.pkl', X_train2, y_train2)
testing('pca_model_name.pkl', X_test2, y_test2)