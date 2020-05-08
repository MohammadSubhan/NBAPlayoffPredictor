import sklearn
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix

def show_results(y_test, predict_test):
    curAccuracy = str(accuracy_score(y_test, predict_test))
    print("Accuracy of current predictions: " + curAccuracy)
    

def load_training_data():
    training_data_2014 = pd.read_csv('datasets/training_data/2014_2015_training_data.csv') 
    training_data_2015 = pd.read_csv('datasets/training_data/2015_2016_training_data.csv')
    training_data_2016 = pd.read_csv('datasets/training_data/2016_2017_training_data.csv')
    training_data_2017 = pd.read_csv('datasets/training_data/2017_2018_training_data.csv') 
    training_data_2018 = pd.read_csv('datasets/training_data/2018_2019_training_data.csv') 
    return [training_data_2014, training_data_2015, training_data_2016, training_data_2017, training_data_2018]

def load_testing_data():
    testing_data_2014 = pd.read_csv('datasets/testing_data/2014_2015_testing_data.csv') 
    testing_data_2015 = pd.read_csv('datasets/testing_data/2015_2016_testing_data.csv') 
    testing_data_2016 = pd.read_csv('datasets/testing_data/2016_2017_testing_data.csv') 
    testing_data_2017 = pd.read_csv('datasets/testing_data/2017_2018_testing_data.csv') 
    testing_data_2018 = pd.read_csv('datasets/testing_data/2018_2019_testing_data.csv') 
    return [testing_data_2014, testing_data_2015, testing_data_2016, testing_data_2017, testing_data_2018]

def train_and_test_on_individual_seasons_and_playoffs(all_training_data, all_testing_data, numSamples):
    for i in range(numSamples):
        #train new MLP on current season of preseason and regular season game stats and outcomes
        training_data = all_training_data[i]
        predictors = sorted(list(set(list(training_data.columns))-set(target_column)))
        mlp = MLPClassifier(hidden_layer_sizes=(22,66,196,66,22), activation='relu', solver='adam', max_iter=800, random_state=777)
        training_data[predictors] = training_data[predictors]/training_data[predictors].max() #normalize values
        X_train = training_data[predictors].values
        y_train = training_data[target_column].values
        mlp.fit(X_train,y_train.ravel())
        #test new MLP on current season of playoff game stats to see prediction of outcomes
        testing_data = all_testing_data[i]
        testing_data[predictors] = testing_data[predictors]/testing_data[predictors].max()
        X_test = testing_data[predictors].values
        y_test = testing_data[target_column].values
        predict_test = mlp.predict(X_test)
        show_results(y_test, predict_test)
        
target_column = ["Top_Team_Won"]
all_training_data = load_training_data()
all_testing_data = load_testing_data() 
numSamples = len(all_testing_data)
train_and_test_on_individual_seasons_and_playoffs(all_training_data, all_testing_data, numSamples)