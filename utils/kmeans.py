# -*- coding: utf-8 -*-

"""
k-menasによるクラスタリングを実施する
Author:
    Yuki-Kumon
Last Update:
    2020-07-09
"""


from sklearn.cluster import KMeans


class Kmeans():

    def __init__(self, k_means_params):
        self.k_means = KMeans(**k_means_params)

    def predict(self, data):
        return self.k_means.fit_predict(data)
