from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default = "browser"


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    first_graph, second_graph = 1, 2
    if noise == 0:
        first_graph, second_graph = 3, 4
    if noise == 10:
        first_graph, second_graph = 5, 6

    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x = np.linspace(-1.2, 2, n_samples)
    f_x = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    dataset_y = f_x + np.random.normal(0, noise, n_samples)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(x), dataset_y, 2 / 3)
    test_x_for_graph = (test_X.to_numpy()).reshape((test_X.shape[0],))
    train_x_for_graph = (train_X.to_numpy()).reshape((train_X.shape[0],))
    fig1 = go.Figure([
        go.Scatter(name=r"$\text{f(x) - noiseless}$", x=x, y=f_x, mode="markers+lines"),
        go.Scatter(name=r"$\text{train samples}$", x=train_x_for_graph, y=train_y, mode="markers"),
        go.Scatter(name=r"$\text{test samples}$", x=test_x_for_graph, y=test_y, mode="markers")]).update_layout(
        title=r"$\text{("f"{first_graph}"r") True Noiseless Model - Train and Test Sets with "
              r"num samples = "f"{n_samples}"r" and noise = "f"{noise}"r"}$",
        xaxis_title=r"$\text{samples}$",
        yaxis_title=r"$\text{prediction and true values of train and test values}$")
    fig1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_X, train_y = np.array(train_X), np.array(train_y)
    average_train_err, average_validation_err = [], []
    degrees = [k for k in range(11)]
    for k in degrees:
        base_estimator = PolynomialFitting(k)
        train_err, validation_err = cross_validate(base_estimator, train_X, train_y, mean_square_error)
        average_train_err.append(train_err)
        average_validation_err.append(validation_err)

    fig2 = go.Figure([go.Scatter(name=r"$\text{train}$", x=degrees, y=average_train_err, mode="markers+lines"),
                      go.Scatter(name=r"$\text{validation}$", x=degrees, y=average_validation_err,
                                 mode="markers+lines")]).update_layout(
        title=r"$\text{("f"{second_graph}"r") Average Training and Validation Errors as a Function of Degree K. "
              r"num samples = "f"{n_samples}"r" and noise = "f"{noise}"r"}$",
        xaxis_title=r"$\text{k}$",
        yaxis_title=r"$\text{average error}$")
    fig2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k = np.argmin(average_validation_err)
    poly = PolynomialFitting(int(k)).fit(train_X, train_y)
    test_err = poly.loss(np.array(test_X), np.array(test_y))

    print("\tnum samples: ", n_samples, ", noise: ", noise)
    print("\tk: ", k)
    print("\ttest error: ", np.round(test_err, 2))
    print("\tvalidation error: ", np.round(np.min(average_validation_err), 2), "\n")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    data_X, data_y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y, test_X, test_y = data_X[:n_samples], data_y[:n_samples], data_X[n_samples:], data_y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    avr_train_err_ridge, avr_validation_err_ridge, avr_train_err_lasso, avr_validation_err_lasso = [], [], [], []
    lambdas = np.linspace(0.001, 2, n_evaluations)
    for lam in lambdas:
        ridge_train_err, ridge_validation_err = \
            cross_validate(RidgeRegression(lam), train_X, train_y, mean_square_error)
        lasso_train_err, lasso_validation_err = \
            cross_validate(Lasso(alpha=lam), train_X, train_y, mean_square_error)

        avr_train_err_ridge.append(ridge_train_err)
        avr_validation_err_ridge.append(ridge_validation_err)
        avr_train_err_lasso.append(lasso_train_err)
        avr_validation_err_lasso.append(lasso_validation_err)

    fig = go.Figure([
        go.Scatter(name=r"$\text{ridge train error}$", x=lambdas, y=avr_train_err_ridge, mode='lines'),
        go.Scatter(name=r"$\text{ridge validation error}$", x=lambdas, y=avr_validation_err_ridge, mode='lines'),
        go.Scatter(name=r"$\text{lasso train error}$", x=lambdas, y=avr_train_err_lasso, mode='lines'),
        go.Scatter(name=r"$\text{lasso validation error}$", x=lambdas, y=avr_validation_err_lasso, mode='lines')
    ]).update_layout(title=r"$\text{(7) Train and Validation Errors of Ridge and Lasso as a Function of lambda}$",
                     xaxis_title=r"$\text{lambda values}$",
                     yaxis_title=r"$\text{train and validation errors}$")
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lambda_for_ridge = lambdas[np.argmin(avr_validation_err_ridge)]
    best_lambda_for_lasso = lambdas[np.argmin(avr_validation_err_lasso)]

    linear_regression = LinearRegression().fit(train_X, train_y)
    ridge_regression = RidgeRegression(best_lambda_for_ridge).fit(train_X, train_y)
    lasso_regression = Lasso(alpha=best_lambda_for_lasso).fit(train_X, train_y)

    linear_regression_test_err = mean_square_error(test_y, linear_regression.predict(test_X))
    ridge_regression_test_err = mean_square_error(test_y, ridge_regression.predict(test_X))
    lasso_regression_test_err = mean_square_error(test_y, lasso_regression.predict(test_X))

    print("\t---------------------Ridge Regression---------------------")
    print("\tbest lambda: ", best_lambda_for_ridge)
    print("\ttest error: ", ridge_regression_test_err, "\n")
    print("\t---------------------Lasso Regression---------------------")
    print("\tbest lambda: ", best_lambda_for_lasso)
    print("\ttest error: ", lasso_regression_test_err, "\n")
    print("\t---------------------Mean Square Error---------------------")
    print("\ttest error for Mean Square Error: ", linear_regression_test_err)


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
