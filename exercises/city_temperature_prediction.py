from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

pio.renderers.default = "browser"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    full_data = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    full_data["DayOfYear"] = (full_data["Date"]).dt.day_of_year
    full_data = full_data[full_data["Temp"] > -50]
    return full_data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    full_data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_data = full_data[full_data["Country"] == "Israel"].copy()
    israel_data["Year"] = israel_data["Year"].astype(str)
    px.scatter(israel_data, x="DayOfYear", y="Temp", color="Year",
               title="(2.1) The average temperature in Israel in different years",
               color_discrete_sequence=px.colors.qualitative.Light24).update_layout(
        xaxis_title="Day Of Year", yaxis_title="Temperature").show()

    px.bar(israel_data.groupby("Month").Temp.agg("std"),
           title="STD of the daily temperatures per month").update_layout(
        xaxis_title="Month", yaxis_title="Std Values").show()

    # Question 3 - Exploring differences between countries
    average_temp = full_data.groupby(["Month", "Country"], as_index=False).agg({"Temp": ["std", "mean"]})
    px.line(x=average_temp["Month"], y=average_temp["Temp", "mean"], color=average_temp["Country"],
            error_y=average_temp["Temp", "std"]).update_layout(
        title="(3) The average monthly temperature, with error bars.", xaxis_title="Month",
        yaxis_title="Mean Temperature").show()

    # Question 4 - Fitting model for different values of `k`
    train_x_israel, train_y_israel, test_x_israel, test_y_israel = \
        split_train_test(israel_data["DayOfYear"], israel_data["Temp"])
    loss = np.zeros(11)
    for k in range(1, 11):
        polynomial_model = PolynomialFitting(k)
        polynomial_model.fit(np.array(train_x_israel), np.array(train_y_israel))
        loss[k] = (round(polynomial_model.loss(np.array(test_x_israel), np.array(test_y_israel)), 2))
        print("for polynomial model of degree ", k, " the loss is : ", loss[k])

    px.bar(x=np.linspace(1, 10, 10).astype(int), y=loss[1:11],
           title="$\\text{Loss Error as a function of the degree of the polynomial model}$").update_layout(
        xaxis_title="$\\text{Degree of the polynomial model}$",
        yaxis_title="$\\text{Test Error}$").show()

    # Question 5 - Evaluating fitted model on different countries
    polynomial_fitting = PolynomialFitting(5)
    polynomial_fitting.fit(np.array(train_x_israel), np.array(train_y_israel))
    countries = ["South Africa", "Jordan", "The Netherlands"]
    arr = []
    for country in countries:
        data = full_data[full_data["Country"] == country]
        arr.append(polynomial_fitting.loss(np.array(data["DayOfYear"]), np.array(data["Temp"])))
    px.bar(x=np.array(countries), y=np.array(arr),
           title="$\\text{STD of the countries over the Israel model}$").update_layout(
        xaxis_title=None, yaxis_title="$\\text{STD}$").show()
