from PIL import Image, ImageFilter
import os
import glob
import re
import random


class Crop_image:
    def __init__(self, read_path, crop_size, upscale_factor):
        self.read_path = read_path
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor

    def save_patch(self):
        image_list = []
        if not os.path.exists('./data_cropped'):
            os.makedirs('./data_cropped')
        if not os.path.exists('./data_cropped/data_highres'):
            os.makedirs('./data_cropped/data_highres')
        if not os.path.exists('./data_cropped/data_lowres'):
            os.makedirs('./data_cropped/data_lowres')

        for filename in glob.glob(self.read_path + 'images/train/*.jpg'):
            im = Image.open(filename)
            image_list.append(im)

        for filename in glob.glob(self.read_path + 'images/val/*.jpg'):
            im = Image.open(filename)
            image_list.append(im)

        for filename in glob.glob(self.read_path + 'images/test/*.jpg'):
            im = Image.open(filename)
            image_list.append(im)

        random.shuffle(image_list)
        image_list_ = image_list

        stepSize = self.crop_size - (self.upscale_factor * 3)
        Serial = 0
        for image in image_list:
            File_name = re.split(r'[.,/]', image.filename)[-2]
            for x in range(0, image.size[0] - self.crop_size, stepSize):
                for y in range(0, image.size[1] - self.crop_size, stepSize):
                    new_img = image.crop((x, y, x + self.crop_size, y + self.crop_size))
                    new_img.save('./data_cropped/data_highres/' + File_name + '_' + str(Serial) + '.jpg')
                    Serial += 1
            Serial = 0

        Serial = 0

        self.crop_size = self.crop_size // self.upscale_factor
        stepSize = stepSize // self.upscale_factor

        for image_ in image_list_:
            File_name = re.split(r'[.,/]', image_.filename)[-2]
            image_ = image_.filter(ImageFilter.GaussianBlur)
            image_ = image_.resize((image_.size[0] // self.upscale_factor, image_.size[1] // self.upscale_factor))
            for x in range(0, image_.size[0] - self.crop_size, stepSize):
                for y in range(0, image_.size[1] - self.crop_size, stepSize):
                    new_img = image_.crop((x, y, x + self.crop_size, y + self.crop_size))
                    new_img.save('./data_cropped/data_lowres/' + File_name + '_' + str(Serial) + '.jpg')
                    Serial += 1
            Serial = 0
