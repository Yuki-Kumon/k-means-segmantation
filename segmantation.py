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

from misc.raw_read import RawRead

from utils.image_utils import ImageUtils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image1_path', default='./data//ortho2b.raw')
    # parser.add_argument('--image2_path', default='./data/newdata/KumonColor/ortho2a.raw')
    parser.add_argument('--rates', default=[1, 1])

    parser.add_argument('--cut_start', default=[2050, 1600])
    parser.add_argument('--cut_size', default=[1000, 1000])

    args = parser.parse_args()

    rates = [int(x) for x in args.rates]
    cut_start = [int(x) for x in args.cut_start]
    cut_size = [int(x) for x in args.cut_size]

    # load image
    image1 = RawRead.read(args.image1_path, rate=rates[0])
    # image2 = RawRead.read(args.image2_path, rate=rates[1])]

    # とりあえず解像度を落としてみる
    image1 = cv2.resize(image1, (300, 300))

    # increase dimention
    image1_edit = ImageUtils.increase_dimension(image1)
    # flatten
    image1_data = ImageUtils.flatten(image1_edit)
