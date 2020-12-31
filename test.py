import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import math
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import os
from utils import *
import PIL
from PIL import Image, ImageFilter

# matplotlib.use("TKAgg")

upscale_factor = 3


def get_model(upscale_factor=3, channels=1):

    inputs = keras.Input(shape=(None, None, channels))

    x = layers.Conv2D(filters=64, kernel_size=5, padding="same", kernel_initializer="Orthogonal", activation="tanh")(inputs)
    x = layers.Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer="Orthogonal", activation="tanh")(x)
    x = layers.Conv2D(filters=channels * (upscale_factor ** 2), kernel_size=3, padding="same", kernel_initializer="Orthogonal", activation="sigmoid")(x)

    outputs = tf.nn.depth_to_space(x, block_size=upscale_factor, data_format='NHWC')

    return keras.Model(inputs, outputs)


if __name__ == "__main__":
    model = get_model()
    model.load_weights("./tmp_BSD500/checkpoint")

    test_path = './SR_testing_datasets/Set14'

    test_img_paths = sorted(
        [
            os.path.join(test_path, fname)
            for fname in os.listdir(test_path)
            if fname.endswith(".png")
        ]
    )

    total_bicubic_psnr = 0.0
    total_test_psnr = 0.0
    total_bicubic_ssim = 0.0
    total_test_ssim = 0.0

    for index, test_img_path in enumerate(test_img_paths):
        img = load_img(test_img_path)
        img_blur = img.copy()
        img_blur = img_blur.filter(ImageFilter.GaussianBlur)
        lowres_input = get_lowres_image(img_blur, upscale_factor)
        save_input(lowres_input, index, "input")
        w = lowres_input.size[0] * 3
        h = lowres_input.size[1] * 3
        highres_img = img.resize((w, h))
        prediction = upscale_image(model, lowres_input)
        lowres_img = lowres_input.resize((w, h))

        lowres_img_arr = img_to_array(lowres_img)
        highres_img_arr = img_to_array(highres_img)
        predict_img_arr = img_to_array(prediction)

        bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
        test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)
        bicubic_ssim = tf.image.ssim(lowres_img_arr, highres_img_arr, max_val=255)
        test_ssim = tf.image.ssim(predict_img_arr, highres_img_arr, max_val=255)

        total_bicubic_psnr += bicubic_psnr
        total_test_psnr += test_psnr
        total_bicubic_ssim += bicubic_ssim
        total_test_ssim += test_ssim

        print("PSNR of low resolution image and high resolution image is %.4f" % bicubic_psnr)
        print("PSNR of predict and high resolution is %.4f" % test_psnr)
        print("SSIM of low resolution image and high resolution image is %.4f" % bicubic_ssim)
        print("SSIM of predict and high resolution is %.4f" % test_ssim)
        plot_results(lowres_img, index, "lowres")
        plot_results(highres_img, index, "highres")
        plot_results(prediction, index, "prediction")

    lens = len(test_img_paths)
    print("Avg. PSNR of lowres images is %.4f" % (total_bicubic_psnr / lens))
    print("Avg. PSNR of reconstructions is %.4f" % (total_test_psnr / lens))
    print("Avg. SSIM of lowres images is %.4f" % (total_bicubic_ssim / lens))
    print("Avg. SSIM of reconstructions is %.4f" % (total_test_ssim / lens))
