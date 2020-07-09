# -*- coding: utf-8 -*-

"""
画像を操作する関数
Author:
    Yuki-Kumon
Last Update:
    2020-07-09
"""


import numpy as np


class ImageUtils():

    def __init__(self):
        pass

    @staticmethod
    def increase_dimension(image, window_size=3):
        '''
        窓関数を用いて画像の次元を増やす
        '''
        assert window_size % 2 == 1
        ex_pix = int((window_size - 1) / 2)
        im_size = image.shape

        edit_image = np.empty([im_size[0] - ex_pix * 2, im_size[1] - ex_pix * 2, window_size**2])
        for i in range(ex_pix, im_size[0] - ex_pix):
            for j in range(ex_pix, im_size[1] - ex_pix):
                edit_image[i - ex_pix, j - ex_pix] = np.ravel(image[i - ex_pix:i + ex_pix + 1, j - ex_pix:j + ex_pix + 1])

        return edit_image

    @staticmethod
    def flatten(image):
        im_shape = image.shape
        return image.reshape([im_shape[0] * im_shape[1], im_shape[2]])

    @staticmethod
    def reshape(image, shape):
        return image.reshape([x for x in shape[:2]])
