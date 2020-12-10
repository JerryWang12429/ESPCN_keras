import tensorflow as tf

import os
import math
import numpy as np
import argparse
import time

from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import PIL

from model import get_model
from utils import scaling, process_target, upscale_image, plot_results, save_input, process_img
from IPython.display import display
from pre_proc import Crop_image
import glob

# for remote image show with X11 on MacOS
matplotlib.use("TKAgg")

# default root Dir traing dataset
root_dir = "./BSR/BSDS500/data/"


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_image', dest='crop_image', type=bool, default=False)
    parser.add_argument('--crop_size', dest='crop_size', type=int, default=51)
    parser.add_argument('--upscale_factor', dest='upscale_factor', type=int, default=3)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=256)
    parser.add_argument('--epoch', dest='epoch', type=int, default=1000)
    return parser


class ESPCN:

    def __init__(self, crop_size, upscale_factor, batch_size, epoch):
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.batch_size = batch_size
        self.epoch = epoch

    def get_train_data(self, root_dir):
        x_files = glob.glob(root_dir + 'data_lowres/*.jpg')
        y_files = glob.glob(root_dir + 'data_highres/*.jpg')

        files_ds = tf.data.Dataset.from_tensor_slices((x_files, y_files))
        train_dataset = files_ds.take(int(0.8 * len(files_ds)))
        valid_dataset = files_ds.skip(int(0.8 * len(files_ds)))

        train_dataset = train_dataset.map(lambda x, y: (process_img(x), process_img(y)))
        valid_dataset = valid_dataset.map(lambda x, y: (process_img(x), process_img(y)))

        print("Load  " + str(len(train_dataset)) + "  Image for training")
        print("Load  " + str(len(valid_dataset)) + "  Image for validation")

        train_dataset = train_dataset.batch(self.batch_size)
        valid_dataset = valid_dataset.batch(self.batch_size)

        return train_dataset, valid_dataset

    # Use TF Ops to process.
    def process_input(self, input):
        input = tf.image.rgb_to_yuv(input)
        last_dimension_axis = len(input.shape) - 1
        y, u, v = tf.split(input, 3, axis=last_dimension_axis)
        return y

    def get_lowres_image(self, img):
        """Return low-resolution image to use as model input."""
        return img.resize(
            (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
            PIL.Image.BICUBIC,
        )


class ESPCNCallback(keras.callbacks.Callback):
    def __init__(self):
        super(ESPCNCallback, self).__init__()
        self.test_img = espcn.get_lowres_image(load_img(test_img_paths[0]))

    # Store PSNR value in each epoch.
    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
        if epoch % 20 == 0:
            prediction = upscale_image(self.model, self.test_img)
            plot_results(prediction, "epoch-" + str(epoch), "prediction")

    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    crop_image = args.crop_image
    crop_size = args.crop_size
    upscale_factor = args.upscale_factor
    batch_size = args.batch_size
    epoch = args.epoch

    dataset = os.path.join(root_dir, "images")
    test_path = os.path.join(dataset, "test")

    if crop_image:
        print('Starting pre-crop image into patch')
        pre_process = Crop_image(root_dir, crop_size, upscale_factor)
        pre_process.save_patch()

    root_dir = './data_cropped/'
    espcn = ESPCN(crop_size, upscale_factor, batch_size, epoch)

    # Scale from (0, 255) to (0, 1)
    train_ds, valid_ds = espcn.get_train_data(root_dir)

    test_img_paths = sorted(
        [
            os.path.join(test_path, fname)
            for fname in os.listdir(test_path)
            if fname.endswith(".jpg")
        ]
    )

    train_ds = train_ds.map(lambda x, y: (scaling(x), scaling(y)))
    valid_ds = valid_ds.map(lambda x, y: (scaling(x), scaling(y)))

    train_ds = train_ds.map(
        lambda x, y: (espcn.process_input(x), process_target(y))
    )

    train_ds = train_ds.prefetch(buffer_size=32)

    valid_ds = valid_ds.map(
        lambda x, y: (espcn.process_input(x), process_target(y))
    )

    valid_ds = valid_ds.prefetch(buffer_size=32)
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=100)
    checkpoint_filepath = "./tmp/checkpoint"

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="loss",
        mode="min",
        save_best_only=True,
    )

    model = get_model()
    model.summary()

    callbacks = [ESPCNCallback(), early_stopping_callback, model_checkpoint_callback]
    loss_fn = keras.losses.MeanAbsoluteError()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer, loss=loss_fn,
    )

    start_time = time.time()

    history = model.fit(train_ds, epochs=epoch, callbacks=callbacks, validation_data=valid_ds, verbose=2)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('Loss.png')
    plt.close()

    print("--- %s seconds ---" % (time.time() - start_time))

    model.load_weights(checkpoint_filepath)

    total_bicubic_psnr = 0.0
    total_test_psnr = 0.0
    total_bicubic_ssim = 0.0
    total_test_ssim = 0.0

    for index, test_img_path in enumerate(test_img_paths):
        img = load_img(test_img_path)
        lowres_input = espcn.get_lowres_image(img)
        save_input(lowres_input, index, "input")
        w = lowres_input.size[0] * upscale_factor
        h = lowres_input.size[1] * upscale_factor
        highres_img = img.resize((w, h))
        prediction = upscale_image(model, lowres_input)
        lowres_img = lowres_input.resize((w, h), PIL.Image.BICUBIC)
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

    print("Avg. PSNR of lowres images is %.4f" % (total_bicubic_psnr / 200))
    print("Avg. PSNR of reconstructions is %.4f" % (total_test_psnr / 200))
    print("Avg. SSIM of lowres images is %.4f" % (total_bicubic_ssim / 200))
    print("Avg. SSIM of reconstructions is %.4f" % (total_test_ssim / 200))
