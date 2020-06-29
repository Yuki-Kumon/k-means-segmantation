# -*- coding: utf-8 -*-

"""
k-menasを用いて画像をnクラスに分類する
Author:
    Yuki-Kumon
Last Update:
    2020-06-29
Reference:
    https://enakai00.hatenablog.com/entry/2015/04/14/181305
"""

import numpy as np
from numpy.random import randint

import cv2


def k_means(image, k=3):
    """
    k-menasを用いて画像をnクラスに分類する
    """

    # 1次元配列に変換しておく
    shape = image.shape
    pixels = np.ravel(image)

    cls = [0] * len(pixels)

    # 代表色の初期値をランダムに設定
    center = []
    for i in range(k):
        center.append(np.array([randint(256)]))
    print (map(lambda x: x.tolist(), center))
    distortion = 0

    # 最大50回のIterationを実施
    for iter_num in range(50):
        center_new = []
        for i in range(k):
            center_new.append(np.array([0]))
        num_points = [0] * k
        distortion_new = 0

        # E Phase: 各データが属するグループ（代表色）を計算
        for pix, point in enumerate(pixels):
            min_dist = 256*256*1
            # point = np.array(point)  # 一次元画像なのでこれは無視
            for i in range(k):
                d = sum([x*x for x in point-center[i]])
                if d < min_dist:
                    min_dist = d
                    cls[pix] = i
            center_new[cls[pix]] += point
            num_points[cls[pix]] += 1
            distortion_new += min_dist

        # M Phase: 新しい代表色を計算
        for i in range(k):
            center_new[i] = center_new[i] / num_points[i]
        center = center_new
        print (map(lambda x: x.tolist(), center))
        print ("Distortion = %d" % distortion_new)

        # Distortion(J)の変化が0.5%未満になったら終了
        if iter_num > 0 and distortion - distortion_new < distortion * 0.005:
            break
        distortion = distortion_new

    # 画像データの各ピクセルを代表色で置き換え
    for pix, point in enumerate(pixels):
        pixels[pix] = tuple(center[cls[pix]])

    # return pixels

    image = pixels.reshape(shape)

    return image


if __name__ == '__main__':
    import agrparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./data/newdata/orthoa.raw')
    args = parser.parse_args()
    # 画像の読み込み
    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)

    # 処理の実行
    image = k_means(image)

    # 結果の書き出し
    cv2.imwrite('./out.png', image)
