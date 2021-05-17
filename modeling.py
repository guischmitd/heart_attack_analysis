import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, r2_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from matplotlib import pyplot as plt
from tqdm import tqdm

def preprocess(df, target_col='target'):
    X = df.drop(target_col, axis=1).copy()
    y = df[[target_col]].copy()
    return X, y


def train(X, y, test_size, model_constructor=LinearRegression, **model_args):
    model = model_constructor(**model_args)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None)
    
    model.fit(X_train, y_train.values.reshape(-1,))

    return model, X_train, X_test, y_train, y_test


def evaluate(model, X_train, X_test, y_train, y_test, metrics=[f1_score, accuracy_score, precision_score, recall_score]):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    score_train, score_test = {}, {}

    if isinstance(metrics, list):
        for metric in metrics:
            score_train[metric.__name__] = metric(y_train, np.clip(y_pred_train.round(), 0, 1))
            score_test[metric.__name__] = metric(y_test, np.clip(y_pred_test.round(), 0, 1))
    else:
        score_train[metrics.__name__] = metrics(y_train, y_pred_train.round())
        score_test[metrics.__name__] = metrics(y_test, y_pred_test.round())


    return score_train, score_test


def optimize(df, test_size, model_constructor, param_list, target_col='target', eval_metric='f1_score', plot=True, **kwargs):
    """
    Optimizes a predictive model based on a list of parameter variations.
    params:
        df - A Dataframe containing all data samples and features
        test_size - float belonging to the open interval (0, 1) to define the test set relative size
        model_constructor - SKLearn model constructor (must implement .fit() and .predict())
        param_list - a list containing at least one dict with keyword arguments for model training
        target_col - string that defines the target column of the classification problem (default is "target")
        eval_metric - A single metric with which to choose the best performing model (default is "f1_score")
        plot - Plots the train/test scores (default is True)

    returns:
        out_model - The best scoring trained model instance
        training_data - A list of dicts containing the parameters used for each training and its respective train and test scores
        best_score - A dict containing the scores achieved by the best trained model
        best_params - The input dict of model parameters that resulted in the best value for `eval_score`
    """
    X, y = preprocess(df, target_col=target_col)
    training_data = []

    out_model = None     
    top_test_score = 0   
    
    for params in tqdm(param_list, desc='Training models...'):
        model, *split_data = train(X, y, test_size=test_size, model_constructor=model_constructor, **params)
        train_score, test_score = evaluate(model, *split_data)
        training_data.append({'params': params, 'train_score': train_score, 'test_score': test_score})

        if test_score[eval_metric] > top_test_score:
            out_model = model
            top_test_score = test_score[eval_metric]
            best_params = params
            best_score = test_score
    
    if plot:
        plt.figure(figsize=kwargs['figsize'] if 'figsize' in kwargs else (20, 10))
        plt.title(eval_metric)
        xlabels = [str(x['params'])+'\n' for x in training_data]
        plt.plot(xlabels, [x['train_score'][eval_metric] for x in training_data])
        plt.plot(xlabels, [x['test_score'][eval_metric] for x in training_data])
        plt.legend(['Train', 'Test'])
        plt.xticks(rotation=90)
        plt.grid('on')
        plt.show()

    return out_model, training_data, best_score, best_params