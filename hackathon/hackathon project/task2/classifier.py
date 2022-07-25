# coding=utf-8
import sys

import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import matplotlib

tumor_areas_labels =['BON - Bones', 'HEP - Hepatic', 'LYM - Lymph nodes', 'SKI - Skin',
                     'PUL - Pulmonary', 'PER - Peritoneum', 'MAR - Bone Marrow',
                     'BRA - Brain', 'ADR - Adrenals', 'OTH - Other', 'PLE - Pleura']


def marge_data_and_labels(data_path, labels_path):
    data, labels = pd.read_csv(data_path), pd.read_csv(labels_path)
    data['labels'] = labels
    return data


def preprocess_data_Q1(data,train=True):
    df = data.drop_duplicates()
    enc = LabelEncoder()
    df['id-hushed_internalpatientid'] = enc.fit_transform(df['id-hushed_internalpatientid'])
    df[' Form Name'] = enc.fit_transform(df[' Form Name'])
    df['אבחנה-Surgery sum'] = df['אבחנה-Surgery sum'].fillna('0')
    df['אבחנה-Nodes exam'] = df['אבחנה-Nodes exam'].fillna('0')
    df['אבחנה-Positive nodes'] = df['אבחנה-Positive nodes'].fillna('0')
    df['אבחנה-Side'] = enc.fit_transform(df['אבחנה-Side'].astype(str))
    df['אבחנה-Histopatological degree'] = enc.fit_transform(df['אבחנה-Histopatological degree'])
    df['אבחנה-Histological diagnosis'] = enc.fit_transform(df['אבחנה-Histological diagnosis'].astype(str))
    df['אבחנה-M -metastases mark (TNM)'] = df['אבחנה-M -metastases mark (TNM)'].fillna('MX')
    df['אבחנה-M -metastases mark (TNM)'] = enc.fit_transform(df['אבחנה-M -metastases mark (TNM)'])
    df['אבחנה-Ivi -Lymphovascular invasion'] = df['אבחנה-Ivi -Lymphovascular invasion'].fillna('none')
    neg = ['-', 'No', 'no', '(-)', 'NO', 'neg', 'not', 'pos']
    df['אבחנה-Ivi -Lymphovascular invasion'] = df['אבחנה-Ivi -Lymphovascular invasion'].apply(
        lambda x: -1 if x in neg else 0 if x == 'none' else -1)
    df['אבחנה-Basic stage'] = enc.fit_transform(df['אבחנה-Basic stage'])
    df['אבחנה-Surgery name1'] = enc.fit_transform(df['אבחנה-Surgery name1'].astype(str))
    df['אבחנה-Surgery name2'] = enc.fit_transform(df['אבחנה-Surgery name2'].astype(str))
    df['אבחנה-Surgery name3'] = enc.fit_transform(df['אבחנה-Surgery name3'].astype(str))
    df['אבחנה-Lymphatic penetration'] = df['אבחנה-Lymphatic penetration'].replace(
        to_replace='Null', value='L0 - No Evidence of invasion')
    df['אבחנה-Lymphatic penetration'] = enc.fit_transform(df['אבחנה-Lymphatic penetration'])
    d = df[[' Form Name',
            'אבחנה-Basic stage',
            'אבחנה-Age',
            'אבחנה-Positive nodes',
            'אבחנה-Nodes exam',
        'אבחנה-Histopatological degree',
            'אבחנה-Side',
        'id-hushed_internalpatientid',
            'אבחנה-Histological diagnosis',
            'אבחנה-M -metastases mark (TNM)',
            'אבחנה-Ivi -Lymphovascular invasion',
            'אבחנה-Surgery name1',
            'אבחנה-Surgery name2',
            'אבחנה-Surgery name3',
            ]]
    d = pd.concat([d.drop('אבחנה-Side', axis=1), pd.get_dummies(d['אבחנה-Side'])], axis=1)
    labels = []
    if train:
        labels = df['labels']
    return d, labels


def split_labels_Q1(y_train_Q1):
    y = []
    for label in tumor_areas_labels:
        arr = []
        for w in y_train_Q1:
            if label in w:
                arr.append(1)
            else:
                arr.append(0)
        y.append((np.array(arr)))
    y= np.array(y)
    return y


def question_Q1(data_path, labels_path, test_path):
    data = marge_data_and_labels(data_path, labels_path)

    x_data_Q1, y_data_Q1 = preprocess_data_Q1(data)

    x_train_Q1, x_tmp_Q1, y_train_Q1, y_tmp_Q1 = train_test_split(x_data_Q1, y_data_Q1, train_size=0.7)
    x_val_Q1, x_test_Q1, y_val_Q1, y_test_Q1 = train_test_split(x_tmp_Q1, y_tmp_Q1, train_size=0.5)

    labels_train_array = split_labels_Q1(y_train_Q1)

    test = pd.read_csv(test_path)
    test_data, not_used = preprocess_data_Q1(test, train=False)

    return_labels = []
    for i, label in enumerate(tumor_areas_labels):
        estimator_Q1 = RandomForestClassifier(n_estimators=85)
        estimator_Q1.fit(x_train_Q1, labels_train_array[i])
        y_pred = estimator_Q1.predict(test_data)
        return_labels.append(y_pred)
    return_labels = np.array(return_labels).T

    y_prediction = []
    for row in return_labels:
        ret = '['
        for col, label in enumerate(tumor_areas_labels):
            if row[col]:
                if ret != '[':
                    ret += ", "
                ret += "'" + label + "'"
        ret += ']'
        y_prediction.append(np.array(ret))
    y_prediction = np.array(y_prediction).T

    (pd.Series(y_prediction)).to_csv('part1/predictions.csv', index=False)

    # (pd.Series(y_prediction)).to_csv(
    #     'Data and Supplementary Material-20220602/Mission 2 - Breast Cancer/y_pred_Q1.csv', index=False)
    # y_test_Q1.to_csv(
    #     'Data and Supplementary Material-20220602/Mission 2 - Breast Cancer/y_true_Q1.csv', index=False)


def preprocess_data_Q2(data, train=True):
    df = data.drop_duplicates()
    enc = LabelEncoder()
    df['id-hushed_internalpatientid'] = enc.fit_transform(df['id-hushed_internalpatientid'].astype(str))
    df['אבחנה-Surgery sum'] = df['אבחנה-Surgery sum'].fillna('0')
    df['אבחנה-Nodes exam'] = df['אבחנה-Nodes exam'].fillna('0')
    df['אבחנה-Positive nodes'] = df['אבחנה-Positive nodes'].fillna('0')
    df['אבחנה-Stage'] = df['אבחנה-Stage'].fillna('Not yet Established')
    df['אבחנה-Stage'] = enc.fit_transform(df['אבחנה-Stage'].astype(str))
    df['אבחנה-T -Tumor mark (TNM)'] = df['אבחנה-T -Tumor mark (TNM)'].fillna('Not yet Established')
    df['אבחנה-T -Tumor mark (TNM)'] = enc.fit_transform(df['אבחנה-T -Tumor mark (TNM)'].astype(str))
    df['אבחנה-N -lymph nodes mark (TNM)'] = df['אבחנה-N -lymph nodes mark (TNM)'].fillna('Not yet Established')
    df['אבחנה-N -lymph nodes mark (TNM)'] = df['אבחנה-N -lymph nodes mark (TNM)'].replace(
        to_replace='?NAME#', value='Not yet Established')
    df['אבחנה-N -lymph nodes mark (TNM)'] = enc.fit_transform(df['אבחנה-N -lymph nodes mark (TNM)'].astype(str))
    df['אבחנה-Histopatological degree'] = enc.fit_transform(df['אבחנה-Histopatological degree'].astype(str))
    df['אבחנה-Histological diagnosis'] = enc.fit_transform(df['אבחנה-Histological diagnosis'].astype(str))
    df['אבחנה-M -metastases mark (TNM)'] = df['אבחנה-M -metastases mark (TNM)'].fillna('MX')
    df['אבחנה-M -metastases mark (TNM)'] = enc.fit_transform(df['אבחנה-M -metastases mark (TNM)'].astype(str))
    df['אבחנה-Ivi -Lymphovascular invasion'] = df['אבחנה-Ivi -Lymphovascular invasion'].fillna('none')
    neg = ['-', 'No', 'no', '(-)', 'NO', 'neg', 'not', 'pos']
    df['אבחנה-Ivi -Lymphovascular invasion'] = df['אבחנה-Ivi -Lymphovascular invasion'].apply(
        lambda x: -1 if x in neg else 0 if x == 'none' else -1)
    df['אבחנה-Surgery sum'] = df['אבחנה-Surgery sum'].fillna(0)
    df['אבחנה-Lymphatic penetration'] = df['אבחנה-Lymphatic penetration'].replace(
        to_replace='Null', value='L0 - No Evidence of invasion')
    df['אבחנה-Lymphatic penetration'] = enc.fit_transform(df['אבחנה-Lymphatic penetration'].astype(str))
    d = df[[
            'אבחנה-Age',
            'אבחנה-Positive nodes',
            'אבחנה-Nodes exam',
        'אבחנה-Histopatological degree',
        'id-hushed_internalpatientid',
            'אבחנה-Histological diagnosis',
            'אבחנה-M -metastases mark (TNM)',
            'אבחנה-Stage',
             'אבחנה-Surgery sum',
        'אבחנה-T -Tumor mark (TNM)',
        'אבחנה-N -lymph nodes mark (TNM)'
            ]]
    labels = []
    if train:
        labels = df['labels']
        # plot graph
        # import matplotlib.pyplot as plt
        # corr = (np.cov(d['אבחנה-Stage'], labels)[0, 1])/(np.std(d['אבחנה-Stage'])*np.std(labels))
        # figure = plt.figure()
        # ax = figure.add_subplot(111)
        # ax.set_xlabel(f"stage values")
        # ax.set_ylabel("tumor size")
        # ax.set_title(f"Correlation: = " + f"{corr}")
        # ax.scatter(d['אבחנה-Stage'], labels, marker='.')
        # corr = (np.cov(d['אבחנה-T -Tumor mark (TNM)'], labels)[0, 1])/(np.std(d['אבחנה-T -Tumor mark (TNM)'])*np.std(labels))
        # figure = plt.figure()
        # ax = figure.add_subplot(111)
        # ax.set_xlabel(f"tumor mark")
        # ax.set_ylabel("tumor size")
        # ax.set_title(f"Correlation: = " + f"{corr}")
        # ax.scatter(d['אבחנה-T -Tumor mark (TNM)'], labels, marker='.')
        # plt.show()
    return d, labels


def question_Q2(data_path, labels_path, test_path):
    data = marge_data_and_labels(data_path, labels_path)
    x_data, y_data = preprocess_data_Q2(data)
    x_train, x_tmp, y_train, y_tmp = train_test_split(x_data, y_data, test_size=0.3,train_size=0.7)
    x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, test_size=0.5,train_size=0.5)

    test = pd.read_csv(test_path)
    test_data, not_used = preprocess_data_Q2(test, train=False)

    estimator = RandomForestRegressor(n_estimators=10)
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(test_data)
    (pd.Series(y_pred)).to_csv('part2/predictions.csv', index=False)

    # (pd.Series(y_pred)).to_csv(
    #     'Data and Supplementary Material-20220602/Mission 2 - Breast Cancer/y_pred_Q2.csv', index=False)
    # y_test.to_csv(
    #     'Data and Supplementary Material-20220602/Mission 2 - Breast Cancer/y_true_Q2.csv', index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Q1
    path = 'Data and Supplementary Material-20220602/Mission 2 - Breast Cancer'
    data_path = path + '/train.feats.csv'
    labels_path_Q1 = path + '/train.labels.0.csv'
    test_path = path + '/test.feats.csv'
    question_Q1(data_path, labels_path_Q1, test_path)

    # Q2
    labels_path_Q2 = path + '/train.labels.1.csv'
    question_Q2(data_path, labels_path_Q2, test_path)
