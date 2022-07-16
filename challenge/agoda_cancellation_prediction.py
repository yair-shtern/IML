import sklearn

from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(filename: str, train=True):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    full_data = pd.read_csv(filename,
                            ).replace("Pay Now", 1). \
        replace("Pay Later", 0).replace("Pay at Check-in", -1) \
        .replace('TRUE', 1).replace('FALSE', 0).replace('True', 1).replace('False', 0).drop_duplicates()

    if train:
        # full_data["cancellation_datetime"] = full_data["cancellation_datetime"].fillna(0)
        full_data = full_data.dropna()
    else:
        full_data = full_data.fillna(0)
    # booking_datetime = full_data["booking_datetime"]

    full_data["days_before_checking"] = (pd.to_datetime(full_data["checkin_date"], dayfirst=True) -
                                         pd.to_datetime(full_data["booking_datetime"], dayfirst=True)).dt.days

    full_data["booking_day"] = pd.to_datetime(full_data["booking_datetime"], dayfirst=True).dt.day
    full_data["booking_day_of_week"] = pd.to_datetime(full_data["booking_datetime"], dayfirst=True).dt.dayofweek
    full_data["day"] = pd.to_datetime(full_data["checkin_date"], dayfirst=True).dt.day
    full_data["day_of_week"] = pd.to_datetime(full_data["checkin_date"], dayfirst=True).dt.dayofweek
    if train == True:
        f = np.zeros(full_data.shape[0])
        booking = np.array(full_data["booking_datetime"])
        canceling = np.array(full_data["cancellation_datetime"])
        for i in range(full_data.shape[0]):
            date = pd.to_datetime(booking[i], errors="coerce", dayfirst=True)
            cancel = pd.to_datetime(canceling[i], errors="coerce", dayfirst=True)
            if cancel != 0 and date != 0 and cancel.month == date.month + 1 and 7 <= cancel.day <= 13:
                f[i] = 1
            else:
                f[i] = 0

        full_data["cancellation_datetime"] = f
        full_data = full_data[full_data['days_before_checking'] >= 0]

    enc = LabelEncoder()
    full_data["customer_nationality"] = enc.fit_transform(full_data["customer_nationality"])
    full_data["split_cancellation_policy_code_1"] = full_data["cancellation_policy_code"].str.split("_").str[
        -1].to_numpy().reshape(
        full_data.shape[0], )
    full_data["split_cancellation_policy_code_1"] = enc.fit_transform(full_data["split_cancellation_policy_code_1"])
    full_data["split_cancellation_policy_code_2"] = full_data["cancellation_policy_code"].str.split("_").str[
        -2].to_numpy().reshape(
        full_data.shape[0], )
    full_data["split_cancellation_policy_code_2"] = enc.fit_transform(full_data["split_cancellation_policy_code_2"])

    full_data["split_cancellation_policy_code_3"] = full_data["cancellation_policy_code"].str.split("_").str[
        -3].to_numpy().reshape(
        full_data.shape[0], )
    full_data["split_cancellation_policy_code_3"] = enc.fit_transform(full_data["split_cancellation_policy_code_3"])

    full_data["cancellation_policy_code"] = enc.fit_transform(full_data["cancellation_policy_code"])
    full_data["accommadation_type_name"] = enc.fit_transform(full_data["accommadation_type_name"])

    full_data["h_customer_id"] = enc.fit_transform(full_data["h_customer_id"])
    full_data["booking_datetime"] = enc.fit_transform(full_data["booking_datetime"])

    full_data["original_payment_type"] = enc.fit_transform(full_data["original_payment_type"])
    full_data["original_payment_method"] = enc.fit_transform(full_data["original_payment_method"])
    full_data["hotel_country_code"] = enc.fit_transform(full_data["hotel_country_code"])
    # full_data["no_of_adults"] = full_data["no_of_adults"].apply(lambda x: 0 if x >= 4 else 1)
    # full_data["no_of_children"] = full_data["no_of_children"].apply(lambda x: 0 if x > 3 else 1)
    # # #     full_data["cancellation_policy_code"] = enc.fit_transform(full_data["cancellation_policy_code"])
    # # #     full_data["no_of_children"] = full_data["no_of_children"].apply(lambda x: 0 if x < 2 else 1)
    # full_data["no_of_room"] = full_data["no_of_children"].apply(lambda x: 0 if x >= 2 else 1)
    # full_data["original_selling_amount"] = full_data["original_selling_amount"].apply(lambda x: 0 if x > 1000 else 1)
    # #     full_data["no_of_adults"] = full_data["no_of_adults"].apply(lambda x: 0 if x > 4 else 1)
    # full_data["days_before_checking"] = full_data["days_before_checking"].apply(lambda x: 0 if x < 20 else 1)
    #
    # book = np.zeros(full_data.shape[0])
    # booking = np.array(full_data["booking_datetime"])
    # for i in range(full_data.shape[0]):
    #     date = pd.to_datetime(booking[i], errors="coerce", dayfirst=True)
    #     book[i] = date.day
    # full_data["booking_datetime"] = book
    full_data["a"] = full_data["request_airport"] \
                     + full_data["request_earlycheckin"] + full_data["request_latecheckin"]
    full_data["n"] = full_data["no_of_extra_bed"] + full_data["no_of_adults"] + \
                     full_data["no_of_children"] \
                     + full_data["no_of_room"]
    # full_data["n"] = full_data["n"].apply(lambda x: 0 if x <= 5 else 1)
    full_data["checkin"] = full_data["request_earlycheckin"] + full_data["request_latecheckin"]
    features = full_data[[
        "checkin",
        # "day",
        # "day_of_week",
        # "booking_day",
        # "booking_day_of_week",
        # "split_cancellation_policy_code_1",
        "split_cancellation_policy_code_2",
        # "split_cancellation_policy_code_3",
        "a",
        "n",
        "no_of_extra_bed",
        "request_nonesmoke",
        "request_latecheckin",
        # "request_highfloor",
        # "request_largebed",
        # "request_twinbeds",
        "request_airport",
        "request_earlycheckin",
        # "hotel_chain_code",
        # "h_booking_id",
        "is_first_booking",
        # "cancellation_policy_code",
        "charge_option",
        # "no_of_adults",
        # "h_customer_id",
        # "hotel_star_rating",
        "guest_is_not_the_customer",
        # "no_of_children",
        "no_of_room",
        # "hotel_area_code",
        # "hotel_brand_code",
        # "hotel_city_code",
        "customer_nationality",
        # "hotel_id",
        "original_payment_type",
        # "original_selling_amount",
        # "original_payment_method",
        # "hotel_country_code",
        "is_user_logged_in",
        "days_before_checking",
        # "booking_datetime",
        "accommadation_type_name"
    ]]

    if train:
        labels = full_data["cancellation_datetime"]
    else:
        labels = []
    return features, labels


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pred = estimator.predict(X)
    print("num positive in estimation = ", np.sum(pred))
    pd.DataFrame(pred, columns=["predicted_values"]).to_csv(filename, index=False)


def load_labels(filename: str):
    labels_and_id = pd.read_csv(filename)
    return labels_and_id['cancel'].astype(int).to_numpy().reshape(
        labels_and_id.shape[0], )


def update_train_data(train_X, train_y, df_test_week1, labels_week1, zero_frac=0.1):
    df_test_week1["labels"] = labels_week1
    zeros = df_test_week1[df_test_week1["labels"] == 0]
    ones = df_test_week1[df_test_week1["labels"] == 1]
    zeros = zeros.sample(frac=zero_frac)
    # ones = pd.concat([ones, ones])
    tmp = pd.concat([zeros, ones])
    train_X["labels"] = train_y
    update_train = pd.concat([train_X, tmp])
    update_train_y = update_train["labels"]
    update_train_X = update_train.drop("labels", axis=1)
    return update_train_X, update_train_y


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    train_X, test_X, train_y, test_y = \
        train_test_split(df, cancellation_labels, test_size=0.15)

    # update week1
    df_test_week1, cancellation_labels_week1 = load_data("datasets/test sets/week_1_test_data.csv", False)
    labels_week1 = load_labels("datasets/labels/test_set_1_labels.csv")

    # update week2
    df_test_week2, cancellation_labels_week2 = load_data("datasets/test sets/week_2_test_data.csv", False)
    labels_week2 = load_labels("datasets/labels/test_set_2_labels.csv")

    # update week3
    df_test_week3, cancellation_labels_week3 = load_data("datasets/test sets/week_3_test_data.csv", False)
    labels_week3 = load_labels("datasets/labels/test_set_3_labels.csv")

    # update week4
    df_test_week4, cancellation_labels_week4 = load_data("datasets/test sets/week_4_test_data.csv", False)
    labels_week4 = load_labels("datasets/labels/test_set_4_labels.csv")

    # update week5
    df_test_week5, cancellation_labels_week5 = load_data("datasets/test sets/week_5_test_data.csv", False)
    labels_week5 = load_labels("datasets/labels/test_set_5_labels.csv")

    # update week6
    df_test_week6, cancellation_labels_week6 = load_data("datasets/test sets/week_6_test_data.csv", False)
    labels_week6 = load_labels("datasets/labels/week_6_labels.csv")

    # update week7
    df_test_week7, cancellation_labels_week7 = load_data("datasets/test sets/week_7_test_set.csv", False)
    labels_week7 = load_labels("datasets/labels/week_7_labels.csv")

    # update week8
    df_test_week8, cancellation_labels_week8 = load_data("datasets/test sets/week_8_test_set.csv", False)
    labels_week8 = load_labels("datasets/labels/week_8_labels.csv")

    # update week9
    df_test_week9, cancellation_labels_week9 = load_data("datasets/test sets/week_9_test_data.csv", False)
    labels_week9 = load_labels("datasets/labels/week_9_labels.csv")

    # update week10
    df_test_week10, cancellation_labels_week10 = load_data("datasets/test sets/week_10_test_data.csv", False)
    labels_week10 = load_labels("datasets/labels/week_10_labels.csv")

    update_train_week1_X, update_train_week1_y = update_train_data(
        train_X, train_y, df_test_week1, labels_week1, 0.15)
    update_train_week2_X, update_train_week2_y = update_train_data(
        update_train_week1_X, update_train_week1_y, df_test_week2, labels_week2, 0.1)
    update_train_week3_X, update_train_week3_y = update_train_data(
        update_train_week2_X, update_train_week2_y, df_test_week3, labels_week3, 0.1)
    update_train_week4_X, update_train_week4_y = update_train_data(
        update_train_week3_X, update_train_week3_y, df_test_week4, labels_week4, 0.05)
    update_train_week5_X, update_train_week5_y = update_train_data(
        update_train_week4_X, update_train_week4_y, df_test_week5, labels_week5, 0.1)
    update_train_week6_X, update_train_week6_y = update_train_data(
        update_train_week5_X, update_train_week5_y, df_test_week6, labels_week6, 0.1)
    update_train_week7_X, update_train_week7_y = update_train_data(
        update_train_week6_X, update_train_week6_y, df_test_week7, labels_week7, 0.15)
    update_train_week8_X, update_train_week8_y = update_train_data(
        update_train_week7_X, update_train_week7_y, df_test_week8, labels_week8, 0.15)
    update_train_week9_X, update_train_week9_y = update_train_data(
        update_train_week8_X, update_train_week8_y, df_test_week9, labels_week9, 0.1)

    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(update_train_week9_X, update_train_week9_y)
    print("Train:")
    # print("train accuracy: ", estimator.loss(update_train_week7_X, update_train_week7_y))
    # print("Week 1:")
    # print("week1 accuracy: ", estimator.loss(df_test_week1, labels_week1))
    # print("Week 2:")
    # print("week2 accuracy: ", estimator.loss(df_test_week2, labels_week2))
    # print("Week 3:")
    # print("week3 accuracy: ", estimator.loss(df_test_week3, labels_week3))
    # print("Week 4:")
    # print("week4 accuracy: ", estimator.loss(df_test_week4, labels_week4))
    # print("Week 5:")
    # print("week5 accuracy: ", estimator.loss(df_test_week5, labels_week5))
    # print("Week 6:")
    # print("week6 accuracy: ", estimator.loss(df_test_week6, labels_week6))
    # print("Week 7:")
    # print("week7 accuracy: ", estimator.loss(df_test_week7, labels_week7))
    # print("Week 8:")
    # print("week8 accuracy: ", estimator.loss(df_test_week8, labels_week8))
    # print("Week 9:")
    # print("week9 accuracy: ", estimator.loss(df_test_week9, labels_week9))
    print("Week 10:")
    print("week10 accuracy: ", estimator.loss(df_test_week10, labels_week10))
    print("Test:")
    print("test accuracy: ", estimator.loss(test_X, test_y))

    # df_test_week10, cancellation_labels_week10 = load_data("datasets/test sets/week_10_test_data.csv", False)

    # Store model predictions over test set
    # evaluate_and_export(estimator, df_test_week10, "318442241_208916221_207557935.csv")
