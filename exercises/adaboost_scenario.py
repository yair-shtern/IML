import numpy as np
from typing import Tuple
from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default = "browser"


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada_boost = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    train_errors, test_errors = [], []
    for t in range(1, n_learners + 1):
        train_errors.append(ada_boost.partial_loss(train_X, train_y, t))
        test_errors.append(ada_boost.partial_loss(test_X, test_y, t))
    go.Figure([go.Scatter(y=train_errors, mode="lines", name=f"train error"),
               go.Scatter(y=test_errors, mode="lines", name=f"test error")],
              layout=go.Layout(title=f"training and test errors as a function of the number of fitted learners",
                               xaxis=dict(title=f"num of learners"),
                               yaxis=dict(title="error"))).show()

    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    # Question 2: Plotting decision surfaces
    if noise == 0:
        T = [5, 50, 100, 250]
        fig = make_subplots(2, 2, subplot_titles=[f"{t} learners" for t in T])
        for index, ensembles_num in enumerate(T):
            predict = lambda g: ada_boost.partial_predict(g, ensembles_num)
            currTraces = [decision_surface(predict, lims[0], lims[1], showscale=False),
                          go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", marker=dict(color=test_y),
                                     showlegend=False)]
            fig.add_traces(currTraces, rows=int(index / 2) + 1, cols=(index % 2) + 1)
        fig.update_layout(title="Decision surfaces for different num of learners")
        fig.show()

    # Question 3: Decision surface of best performing ensemble
    if noise == 0:
        losses = [ada_boost.partial_loss(test_X, test_y, ensembles_num) for ensembles_num in range(n_learners)]
        predict = lambda g: ada_boost.partial_predict(g, int(np.argmin(losses)))
        go.Figure([decision_surface(predict, lims[0], lims[1], showscale=False),
                   go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", marker=dict(color=test_y),
                              showlegend=False)]).update_layout(
            title=f"num of ensemble = {int(np.argmin(losses))}, accuracy = {1 - losses[int(np.argmin(losses))]}").show()

    # Question 4: Decision surface with weighted samples
    D = 10 * (ada_boost.D_ / np.max(ada_boost.D_))
    if noise == 0:
        D *= 1.5
    go.Figure([decision_surface(ada_boost.predict, lims[0], lims[1], showscale=False),
               go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                          marker=dict(color=train_y.astype(int), colorscale=[custom[0], custom[-1]],
                                      line=dict(color="black", width=1), size=D))]).update_layout(
        title=f"Decision surface with weighted samples").show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
