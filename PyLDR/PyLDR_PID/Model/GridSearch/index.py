from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, KFold
from sklearn.base import clone
from sklearn.metrics import r2_score
import numpy as np


class GridSearch:
    def __init__(self, model, params, xData, yData, k=5, test_size=0.20, temporal=True):
        self.model  = clone(model)
        self.params = params
        self.xData  = xData
        self.yData  = yData
        self.test_size = test_size
        self.k      = k
        self.grid   = None
        self.temporal = temporal
    
    def update(self):
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.xData, self.yData, test_size=self.test_size, shuffle=(False if self.temporal else True))
        cv_splitter = TimeSeriesSplit(n_splits=self.k) if self.temporal else KFold(n_splits=self.k, shuffle=True, random_state=42)
        self.grid   = GridSearchCV(
            estimator=self.model, 
            param_grid=self.params, 
            cv=cv_splitter, 
            scoring='r2', 
            n_jobs=-1, 
            verbose=1,
            return_train_score=False
        )
        
        self.grid.fit(self.xTrain, self.yTrain)

    def evaluate(self):
        model = self.grid.best_estimator_
        yPred = model.predict(self.xTest)
        
        r2 = r2_score(self.yTest, yPred)
        n  = len(self.yTest)
        p  = self.xTest.shape[1]
        
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2 
        return (model, r2_adj)