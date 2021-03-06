# -*- coding: utf-8 -*-

"""
k-menasを用いて画像をnクラスに分類する
次元を増やすため3×3の窓を導入する
Author:
    Yuki-Kumon
Last Update:
    2020-07-09
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from misc.raw_read import RawRead

from utils.image_utils import ImageUtils
from utils.kmeans import Kmeans


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image1_path', default='./data/ortho2b.raw')
    # parser.add_argument('--image2_path', default='./data/newdata/KumonColor/ortho2a.raw')
    parser.add_argument('--rates', default=[1, 1])
    parser.add_argument('--cut_start', default=[2050, 1600])
    parser.add_argument('--cut_size', default=[1000, 1000])
    parser.add_argument('--n_clusters', default=3)

    args = parser.parse_args()

    rates = [int(x) for x in args.rates]
    cut_start = [int(x) for x in args.cut_start]
    cut_size = [int(x) for x in args.cut_size]

    k_means_params = {
        'n_clusters': int(args.n_clusters),
        'n_jobs': -1
    }

    # load image
    image1 = RawRead.read(args.image1_path, rate=rates[0])
    # image2 = RawRead.read(args.image2_path, rate=rates[1])]

    # crop images
    image1 = image1[cut_start[0]:cut_start[0] + cut_size[0], cut_start[1]:cut_start[1] + cut_size[1]]
    # image2 = image2[cut_start[0]:cut_start[0] + cut_size[0], cut_start[1]:cut_start[1] + cut_size[1]]

    cv2.imwrite('./output/origin.png', image1)

    # increase dimention
    image1_edit = ImageUtils.increase_dimension(image1)
    # flatten
    image1_data = ImageUtils.flatten(image1_edit)

    # load kmenas
    kmeans = Kmeans(k_means_params)
    # predict
    res1 = kmeans.predict(image1_data)

    # 二次元配列に戻す
    image1_result = ImageUtils.reshape(res1, [x - 2 for x in cut_size])

    # matplotlibで描画
    y = range(cut_size[0] - 2)
    x = range(cut_size[1] - 2)
    xx, yy = np.meshgrid(x, y)

    plt.axes().set_aspect('equal', 'datalim')
    plt.contourf(xx, yy, image1_result)
    plt.savefig('./output/result.png', dpi=500)
