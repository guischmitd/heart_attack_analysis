import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, r2_score, f1_score, accuracy_score, precision_score, recall_score
from matplotlib import pyplot as plt
from tqdm import tqdm

# Default seed for random processes (splitting and model initialization)
np.random.seed(42)

def preprocess(df, test_size=0.30, target_col='target'):
    """
    Description:
        Splits target and feature columns of a dataframe into X and y dfs or arrays.
        Also splits the data into train and test sets, according to a test_size parameter.

    Arguments:
        df : Pandas dataframe containing all feature columns and target
        test_size : float belonging to the open interval (0, 1) to define the test set relative size
        target_col : string that defines the target column of the classification problem (default is "target")

    Returns:
        X_train : Portion of the dataframe containing all training samples for feature columns
        X_test : Portion of dataframe containing all testing samples for feature columns
        y_train : Portion of dataframe containing all trainig targets
        y_test : Portion of dataframe containing all testing targets
    """
    X = df.drop(target_col, axis=1).copy()
    y = df[[target_col]].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def train(X_train, y_train, model_constructor=LinearRegression, **model_args):
    """
    Description:
        Trains a sklearn compatible model given train and test data as well as a model
        constructor and keyword parameters.

    Arguments:
        X_train : Training set features
        y_train : Training set targets
        model_constructor : sklearn compatible model constructor (must implement .fit() method)
        model_args : dict of keyword arguments passed directly to model_constructor during instantiation

    Returns:
        model : The trained model
    """
    model = model_constructor(**model_args)
    
    model.fit(X_train, y_train.values.reshape(-1,))

    return model


def evaluate(model, X_train, X_test, y_train, y_test, metrics=[f1_score, accuracy_score, precision_score, recall_score]):
    """
    Description:
        Evaluates a trained model given its train and test arrays, 
        as well as sklearn compatible metric functions.

    Arguments:
        model : a trained sklearn compatible model (must implement the .predict() method)
        X_train : Portion of the dataframe containing all training samples for feature columns
        X_test : Portion of dataframe containing all testing samples for feature columns
        y_train : Portion of dataframe containing all trainig targets
        y_test : Portion of dataframe containing all testing targets
        metrics : list of sklearn compatible metric functions with which to evaluate the model
                  (default = [f1_score, accuracy_score, precision_score, recall_score])

    Returns:
        score_train : Metrics dictionary for train scores.
        score_test : Metrics dictionary for test scores.
    """
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
    Description: Optimizes a predictive model based on a list of parameter variations.
    
    Arguments:
        df : A Dataframe containing all data samples and features
        test_size : float belonging to the open interval (0, 1) to define the test set relative size
        model_constructor : SKLearn model constructor (must implement .fit() and .predict())
        param_list : a list containing at least one dict with keyword arguments for model training
        target_col : string that defines the target column of the classification problem (default is "target")
        eval_metric : A single metric with which to choose the best performing model (default is "f1_score")
        plot : Boolean indicating whether to plot the train/test scores during training (default is True)

    Returns:
        best_model - The best scoring trained model instance
        training_history - A list of dicts containing the parameters used for each training and its respective train and test scores
        best_score - A dict containing the scores achieved by the best trained model
        best_params - The input dict of model parameters that resulted in the best value for `eval_score`
    """
    X_train, X_test, y_train, y_test = preprocess(df, test_size=test_size, target_col=target_col)
    training_history = []

    best_model = None     
    top_test_score = 0 
    
    for params in tqdm(param_list, desc='Training models...'):
        model = train(X_train, y_train, model_constructor=model_constructor, **params)
        train_score, test_score = evaluate(model, X_train, X_test, y_train, y_test)
        training_history.append({'params': params, 'train_score': train_score, 'test_score': test_score})

        if test_score[eval_metric] > top_test_score:
            best_model = model
            top_test_score = test_score[eval_metric]
            best_params = params
            best_score = test_score
    
    if plot:
        plt.figure(figsize=kwargs['figsize'] if 'figsize' in kwargs else (20, 10))
        plt.title(eval_metric)
        xlabels = [str(x['params'])+'\n' for x in training_history]
        plt.plot(xlabels, [x['train_score'][eval_metric] for x in training_history])
        plt.plot(xlabels, [x['test_score'][eval_metric] for x in training_history])
        plt.legend(['Train', 'Test'])
        plt.xticks(rotation=90)
        plt.grid('on')
        plt.show()

    return best_model, training_history, best_score, best_params


def plot_score_comparison(scores, model_names, save_path=None):
    """
    Description:
        Plots a score comparison bar plot when provided with score dicts and model_names. 
        The save_path optional argument can be set so the plot is saved at the specified location.

    Arguments:
        scores : list of score dicts (like the ones generated by the evaluate function)
        model_names : list of string names for each model. Must have same length as the `scores` list
        save_path : *Optional*. Saves the generated plot to the provided path if set to a valid string path.

    Returns:
        None
    """
    performance_df = pd.DataFrame(scores, index=model_names).T * 100
    print(performance_df)
    
    (performance_df).plot.bar()
    plt.xticks(rotation=0)
    plt.legend(loc=4)
    plt.grid('on')

    plt.title('Model Scores Comparison (test only)')
    plt.ylabel('Score (%)')
    
    if save_path:
        plt.savefig(save_path)