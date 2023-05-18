import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.model_selection.cross_validate import cross_validate
from sklearn.metrics import roc_curve, auc
from IMLearn.metrics.loss_functions import mean_square_error, misclassification_error
import plotly.graph_objects as go
import plotly.io as pio


# pio.renderers.default = "browser"


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    weight_arr, loss_arr = [], []

    def callback(output=None, weights=None, val=None, grad=None, t=None,
                 eta=None, delta=None):
        weight_arr.append(weights)
        loss_arr.append(val)

    return callback, weight_arr, loss_arr


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    print("\t=========================== Question 4 ===========================")
    fig = go.Figure().update_layout(title="L1 And L2 Convergence Rate",
                                    xaxis_title="num iterations", yaxis_title="norm value")
    for eta in etas:
        curr_eta = FixedLR(eta)
        l1, l2 = L1(init), L2(init)
        callback_l1, weight_l1, loss_l1 = get_gd_state_recorder_callback()
        callback_l2, weight_l2, loss_l2 = get_gd_state_recorder_callback()

        GD_l1 = GradientDescent(learning_rate=curr_eta, callback=callback_l1, out_type="best")
        lowest_l1 = GD_l1.fit(f=l1, X=np.nan, y=np.nan)
        GD_l2 = GradientDescent(learning_rate=curr_eta, callback=callback_l2, out_type="best")
        lowest_l2 = GD_l2.fit(f=l2, X=np.nan, y=np.nan)

        print(f"\t\tmin loss for L1 Module for eta = {eta} is {np.abs(0 - L1(lowest_l1).compute_output())}.\n")
        print(f"\t\tmin loss for L2 Module for eta = {eta} is {np.abs(0 - L2(lowest_l2).compute_output())}.\n")

        fig.add_scatter(y=loss_l1, name=f"L1 for eta = {eta}")
        fig.add_scatter(y=loss_l2, name=f"L2 for eta = {eta}")

        if eta == 0.01:
            plot_descent_path(module=L1, descent_path=np.array(weight_l1),
                              title="For L1 Norm").show()
            plot_descent_path(module=L2, descent_path=np.array(weight_l2),
                              title="For L2 Norm").show()
    fig.show()
    print("\t===================================================================\n")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig = go.Figure().update_layout(title="Convergence Rate For L1 With difference gamma values",
                                    xaxis_title="num iterations", yaxis_title="norm value")
    min_learning_rate = float('inf')
    best_gamma = 0
    for gamma in gammas:
        callback, weights, losses = get_gd_state_recorder_callback()
        curr_learning_rate = ExponentialLR(eta, gamma)
        gradient_descent = GradientDescent(learning_rate=curr_learning_rate, callback=callback)
        l1 = L1(init)
        lowest_learning_rate = np.abs(0 - L1(gradient_descent.fit(f=l1, X=np.nan, y=np.nan)).compute_output())
        if lowest_learning_rate < min_learning_rate:
            min_learning_rate, best_gamma = lowest_learning_rate, gamma
        fig.add_scatter(y=losses, name=f"gamma={gamma}")
    fig.show()
    print("\t=========================== Question 6 ===========================")
    print(f"\t\tThe Lowest L1 Loss Rate for gamma={best_gamma}\n "
          f"\t\tand eta={eta}, is {min_learning_rate}.")
    print("\t===================================================================\n")

    # Plotting the 2D Contour
    curr_eta = ExponentialLR(eta, 0.95)
    callback_l1, weight_l1, loss_l1 = get_gd_state_recorder_callback()
    callback_l2, weight_l2, loss_l2 = get_gd_state_recorder_callback()
    GD_l1 = GradientDescent(learning_rate=curr_eta, callback=callback_l1, out_type="best")
    GD_l1.fit(f=L1(init), X=np.nan, y=np.nan)
    GD_l2 = GradientDescent(learning_rate=curr_eta, callback=callback_l2, out_type="best")
    GD_l2.fit(f=L2(init), X=np.nan, y=np.nan)
    plot_descent_path(module=L1, descent_path=np.array(weight_l1), title="for L1").show()
    plot_descent_path(module=L2, descent_path=np.array(weight_l2), title="for L2").show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    # Plotting convergence rate of logistic regression over SA heart disease data
    logistic_regression = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000))
    logistic_regression.fit(np.array(X_train), np.array(y_train))
    FP_rate, TP_rate, thresholds = roc_curve(y_train, logistic_regression.predict_proba(np.array(X_train)))

    fig = go.Figure().update_layout(title=f"ROC curve Of Fitted Model - AUC = {auc(FP_rate, TP_rate)}",
                                    xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(color="black", dash='dash'), name="Random Model")
    fig.add_scatter(x=FP_rate, y=TP_rate, mode='markers+lines', text=thresholds,
                    showlegend=False, marker_size=5, marker_color="rgb(49,54,149)",
                    hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")
    fig.show()

    alpha = thresholds[np.argmax(TP_rate - FP_rate)]
    logistic_regression_ = LogisticRegression(alpha=alpha).fit(np.array(X_train), np.array(y_train))
    loss = logistic_regression_.loss(np.array(X_test), np.array(y_test))
    print("\t=========================== Question 9 ===========================")
    print(f"\t\tBest alpha for the logistic regression is {alpha}\n"
          f"\t\twith loss of {loss} on the test set.")
    print("\t===================================================================\n")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    l1_lambda, l2_lambda = 0, 0
    l1_lambda_val_score, l2_lambda_val_score = float('inf'), float('inf')
    for curr_lambda in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
        l1 = LogisticRegression(penalty="l1", alpha=0.5, lam=curr_lambda)
        l2 = LogisticRegression(penalty="l2", alpha=0.5, lam=curr_lambda)
        l1_train, l1_val = cross_validate(l1, np.array(X_train), np.array(y_train), scoring=mean_square_error)
        l2_train, l2_val = cross_validate(l2, np.array(X_train), np.array(y_train), scoring=mean_square_error)
        if l1_lambda_val_score >= l1_val:
            l1_lambda_val_score, l1_lambda = l1_val, curr_lambda
        if l2_lambda_val_score >= l2_val:
            l2_lambda_val_score, l2_lambda = l2_val, curr_lambda

    print(l1_val)
    print(l2_val)
    l1_model = LogisticRegression(penalty="l1", alpha=0.5, lam=l1_lambda)
    l1_model.fit(np.array(X_train), np.array(y_train))
    l1_test_loss = l1_model.loss(np.array(X_test), np.array(y_test))

    l2_model = LogisticRegression(penalty="l2", alpha=0.5, lam=l2_lambda)
    l2_model.fit(np.array(X_train), np.array(y_train))
    l2_test_loss = l2_model.loss(np.array(X_test), np.array(y_test))
    print("\t=========================== Question 10 ===========================")
    print(f"\t\tthe best Lambda is {l1_lambda} with test score of {l1_test_loss}.")
    print("\t===================================================================\n")

    print("\t=========================== Question 11 ===========================")
    print(f"\t\tthe best Lambda is {l2_lambda} with test score of {l2_test_loss}.")
    print("\t===================================================================\n")


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
