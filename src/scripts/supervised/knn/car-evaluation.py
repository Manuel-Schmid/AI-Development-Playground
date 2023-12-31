import pickle
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing
import os
from src.config.definitions import ROOT_DIR


# Data Preparation
data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'car-evaluation.data'))

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))  # transforms non-numeric col. values into numpy array of numeric values
maintenance = le.fit_transform(list(data["maintenance"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"  # column to predict

x = list(zip(buying, maintenance, door, persons, lug_boot, safety))  # create new matrix with columns
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

mode = input('Do you want to train a new model (T) or run predictions with the saved model (R)? : ').upper()


# Model Training
if mode == 'T':
    model = KNeighborsClassifier(7)  # parameter is the amount of neighbours ("K")

    model.fit(x_train, y_train)  # train the model
    acc = model.score(x_test, y_test)  # test model accuracy
    print(acc)  # print model accuracy
    with open(os.path.join(ROOT_DIR, 'models', 'car-evaluation-model.pickle'), "wb") as f:  # save trained model in pickle file
        pickle.dump(model, f)


# Print model predictions
if mode == 'R':
    pickle_in = open(os.path.join(ROOT_DIR, 'models', 'car-evaluation-model.pickle'), "rb")  # load trained model from pickle file
    model = pickle.load(pickle_in)

    names = ["unacceptable", "acceptable", "good", "very good"]  # labels to map class predictions > output readability
    predicted = model.predict(x_test)
    for i in range(len(predicted)):
        print("Data: ", x_test[i], "\t | Predicted: ", names[predicted[i]], "\t\t\t\t | Actual: ", names[y_test[i]])
        neighbours = model.kneighbors([x_test[i]], 7, True)
        # print("Neighbours: ", neighbours)


