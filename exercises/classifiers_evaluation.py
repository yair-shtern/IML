import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi

pio.templates.default = "simple_white"
pio.renderers.default = "browser"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 # ("Linearly Inseparable", "linearly_inseparable.npy")
                 ]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def loss_callback(fit: Perceptron, x: np.ndarray, i: int):
            losses.append(fit.loss(x, y))

        Perceptron(callback=loss_callback).fit(X, y)

        # Plot figure of loss as function of fitting iteration
        go.Figure([go.Scatter(y=losses, mode="lines")],
                  layout=go.Layout(title=f"perceptron algorithm's training loss as a function of the training "
                                         f"iterations - in {n} data", xaxis=dict(title="iteration num"),
                                   yaxis=dict(title="loss"))).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        gnb = GaussianNaiveBayes().fit(X, y)
        gnb_pred = gnb.predict(X)

        lda = LDA().fit(X, y)
        lda_pred = lda.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[f"Gaussian Naive Bayes, accuracy = {accuracy(y, gnb_pred)}",
                                            f"Linear Discriminant Analysis, accuracy = {accuracy(y, lda_pred)}"])
        # Add traces for data-points setting symbols and colors
        fig.add_traces([
            go.Scatter(x=X.T[0], y=X.T[1], mode='markers',
                       marker=dict(color=gnb_pred, symbol=y, line=dict(color="black", width=1))),
            go.Scatter(x=X.T[0], y=X.T[1], mode='markers',
                       marker=dict(color=lda_pred, symbol=y, line=dict(color="black", width=1)))],
            rows=[1, 1], cols=[1, 2])

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_traces([
            go.Scatter(x=gnb.mu_.T[0], y=gnb.mu_.T[1], mode='markers',
                       marker=dict(color="black", symbol="x", size=15, line=dict(color="black"))),
            go.Scatter(x=lda.mu_.T[0], y=lda.mu_.T[1], mode='markers',
                       marker=dict(color="black", symbol="x", size=15, line=dict(color="black")))],
            rows=[1, 1], cols=[1, 2])

        # Add ellipses depicting the covariances of the fitted Gaussians
        for c in gnb.classes_:
            fig.add_trace(get_ellipse(gnb.mu_[c], np.diag(gnb.vars_[c])), row=1, col=1)

        for c in lda.classes_:
            fig.add_trace(get_ellipse(lda.mu_[c], lda.cov_), row=1, col=2)
        fig.update_layout(title=f"dataset {f}", showlegend=False)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
