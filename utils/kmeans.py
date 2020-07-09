# -*- coding: utf-8 -*-

"""
k-menasによるクラスタリングを実施する
Author:
    Yuki-Kumon
Last Update:
    2020-07-09
"""


import sklearn


class Kmeans():

    def __init__(self, k_means_params):
        self.k_means = sklearn.cluster.KMeans(**k_means_params)
