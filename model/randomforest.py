import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from Config import Config
from numpy import *
import random
num_folds = 0
seed =0
# Data
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class RandomForest(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = [RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample') for i in Config.TYPE_COLS] 
        self.predictions = None
        self.data_transform()
#train here for many models
    def train(self, data) -> None:
        for i in range(len(self.mdl)):
            self.mdl[i].fit(data.X_train, data.y_train.iloc[:, i])

        #self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        self.predictions = []
        for model in self.mdl:
            predictions = model.predict(X_test)
            self.predictions.append(predictions)
        
        #self.predictions = [model.predict(X_test) for model in self.mdl]
    def print_results(self, data):
        self.predictions = np.array(self.predictions).T
        df = pd.DataFrame(self.predictions)
        # data.df[Config.TYPE_COLS] == df
        


         # Calculate and print accuracy for each type
        #print("============================================== printing accuracy of each depentednt variable  =========================")
        #accuracies = []
        # for i, col in enumerate(Config.TYPE_COLS):
        #     acc = accuracy_score(data.y_test.iloc[:, i], self.predictions[:, i])
        #     accuracies.append(acc)
        #     print(f'Accuracy for {col}: {acc:.2f}')
        
        # Calculate and print cumulative accuracies
        y2_accuracy = accuracy_score(data.y_test.iloc[:, 0], self.predictions[:, 0])
        y2_y3_accuracy = accuracy_score(data.y_test.iloc[:, :2].values.flatten(), self.predictions[:, :2].flatten())
        y2_y3_y4_accuracy = accuracy_score(data.y_test.iloc[:, :3].values.flatten(), self.predictions[:, :3].flatten())
        print("============================================== printing accuracy of depentednt variable after adding output ================================")
        print(f'Accuracy for y2 or Type 2: {y2_accuracy:.2f}')
        print(f'Accuracy for y2 + y3 of Type 2 + Type 3: {y2_y3_accuracy:.2f}')
        print(f'Accuracy for y2 + y3 + y4 of Type 2 + Type 3 + type 4: {y2_y3_y4_accuracy:.2f}')
        print("============================================== End printing accuracy of depentednt variable after adding output ============================")

        for i in range(len(Config.TYPE_COLS)):
            true_classes = data.y_test.iloc[:, i].tolist()
            predicted_classes = self.predictions[:, i].tolist()
            accuracy = accuracy_score(data.y_test.iloc[:, i], self.predictions[:, i]) * 100
            print("============================================== predicted class, true classes and accuracy of y2,y3,y4 ==================================")
            print(f'Predicted Classes for column ----- {Config.TYPE_COLS[i]}: {predicted_classes}')
            print(f'True Classes for column -----  {Config.TYPE_COLS[i]}: {true_classes}')
            print(f'Accuracy for column -----  {Config.TYPE_COLS[i]}: {accuracy:.2f}%')
    
        true_classes = data.y_test.values.tolist()
        predicted_classes = self.predictions.tolist()
        for i in range(len(data.y_test)):
            # Calculate the percentage of correct predictions for each instance
            correct_predictions = np.sum(np.array(true_classes[i]) == np.array(predicted_classes[i]))
            total_predictions = len(true_classes[i])
            percentage_accuracy = (correct_predictions / total_predictions) * 100
            print("============================================== true classes and accuracy of y2,y3,y4 ==================================")
            print(f'Predicted Classes: {predicted_classes[i]}')
            print(f'True Classes: {true_classes[i]}')
            print(f'Accuracy: {percentage_accuracy:.2f}%')

        # Calculate overall accuracy
        overall_accuracy = accuracy_score(data.y_test.values.flatten(), self.predictions.flatten()) * 100
        print(f'Overall Accuracy of my data: {overall_accuracy:.2f}%')


    def data_transform(self) -> None:
        ...

