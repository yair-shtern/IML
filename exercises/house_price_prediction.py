from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"
pio.renderers.default = "browser"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    full_data = pd.read_csv(filename).dropna().drop_duplicates()
    full_data = full_data[full_data["price"] > 0]
    full_data = full_data[full_data["sqft_lot"] > 0]
    full_data = full_data[full_data["bedrooms"] > 0]
    full_data = full_data[full_data["sqft_living"] > 0]
    full_data = full_data[full_data["floors"] > 0]

    full_data["bedrooms_for_floor"] = full_data["bedrooms"] / full_data["floors"]

    full_data = full_data.drop("lat", axis=1)
    full_data = full_data.drop("long", axis=1)
    full_data = full_data.drop("id", axis=1)
    full_data = full_data.drop("date", axis=1)
    full_data["zipcode"] = full_data["zipcode"].astype(int)
    full_data = pd.get_dummies(full_data, prefix='zipcode_', columns=["zipcode"])

    return full_data.drop("price", axis=1), full_data["price"]


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for x in X:
        corr = (np.cov(X[x], y)[0, 1]) / (np.std(X[x]) * np.std(y))
        go.Figure([go.Scatter(x=X[x], y=y, mode='markers')]) \
            .update_layout(title=f"corr: {corr}", xaxis_title=f"{x}",
                           yaxis_title="$\\text{house price}$").write_image(output_path + str(x) + ".png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    features, labels = load_data('../datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(features, labels, "../Ex's/Ex2/Correlation/")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(features, labels)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    mean_loss = []
    std_loss = []
    p_arr = [p for p in range(10, 101)]
    for p in p_arr:
        loss = []
        for i in range(10):
            estimator = LinearRegression()
            samples_X = train_X.sample(frac=p / 100)
            estimator.fit(np.array(samples_X), train_y[samples_X.index])
            loss.append(estimator.loss(np.array(test_X), np.array(test_y)))
        mean_loss.append(np.array(loss).mean())
        std_loss.append(np.std(loss))
    go.Figure(
        [go.Scatter(x=p_arr, y=np.array(mean_loss), mode="markers+lines",
                    name="Mean loss", line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
         go.Scatter(x=p_arr, y=np.array(mean_loss) - 2 * np.array(std_loss), fill=None, mode="lines",
                    line=dict(color="lightgrey"), showlegend=False),
         go.Scatter(x=p_arr, y=np.array(mean_loss) + 2 * np.array(std_loss), fill='tonexty', mode="lines",
                    line=dict(color="lightgrey"), showlegend=False)]).update_layout(
        title="Mean loss as a function of p%, as well as a confidence interval",
        xaxis_title="Percentage", yaxis_title="Mean Square Error").show()
