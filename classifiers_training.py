import numpy
import os

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import numpy as np
from skimage.io import imread
import scipy as sp

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC

letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

names = ["Nearest Neighbors", "Linear SVM", "Decision Tree",
         "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel='linear', probability=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

def read_training_data(train_dir):
    image_data = []
    target_data = []
    for each_letter in letters:
        for each in range(10):
            image_path = os.path.join(train_dir, each_letter, each_letter + "_" + str(each) + '.jpg')
            # read each image of each char
            image_details = imread(image_path)
            # Converted into a gray scale
            image_details = rgb2gray(image_details)

            binary_image = image_details < threshold_otsu(image_details)
            # we need to convert 2d array to 1d because ml classifier require
            # 1d array of each sample

            flat_bin_image = binary_image.reshape(-1)
            image_data.append(flat_bin_image)
            target_data.append(each_letter)

    return (np.array(image_data), np.array(target_data))

# Getting dataset directory
training_dataset_dir = os.path.join(os.getcwd(), 'input')

# Getting training data into vars
image_data, target_data = read_training_data(training_dataset_dir)

# Directory to save models in
models_dir = os.path.join(os.getcwd(), 'output')

# Num_of_fold cross validation - if 4, 3/4 of dataset is for training and 1/4 of data is for testing
num_of_fold = 4

linearSVCResult = []
NeuralNet =[]
if os.path.exists(models_dir):
    for name, model in zip(names, classifiers):

        # accuracy for current model
        accuracy_result = cross_val_score(model, image_data, target_data, cv=num_of_fold)
        print("cross validation result for ", str(num_of_fold), " -fold for " + name)
        print(accuracy_result * 100)
        print("mean for " +  name + " is " + str(numpy.mean(accuracy_result) * 100))
        print("std for " + name + " is " + str(numpy.std(accuracy_result) *100))

        # training the model
        model.fit(image_data, target_data)

        # saving trained data
        save_dir = os.path.join(models_dir, name)
        joblib.dump(model, save_dir + ".pkl")

        if name == 'Linear SVM':
            linearSVCResult = accuracy_result

        if name == 'Neural Net':
            NeuralNet = accuracy_result


# Statistical validation
wilcoxon = sp.stats.wilcoxon(linearSVCResult, NeuralNet)

print("Wilcoxon result: ")
print(wilcoxon)


#neural net i svc zaleznosc statystyczna
#wilcoxon = sp.stats.wilcoxon()
#scipy.stats.t
#scipy stats wilcockson
#zaleznosc statystyczna


# sala 16/17 sekretariat pólka zostawic artkul kolo stolika na plaszcz
