import os
import numpy as np

from glob import glob
import shutil

import datetime

import argparse
import sys

import keras
import efficientnet

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='person classification test code')

    parser.add_argument('--trained_model', default='./models/efficientnet_base.h5', type=str, help='trained keras model path .h5 file')
    parser.add_argument('--test_folder', default='./imgs', type=str, help='directory containing test imgs')
    parser.add_argument('--csv_folder', default='./csvs', type=str, help='directory containing candidate csvs')

    return parser.parse_args(argv)

def make_batch(rgb, image_size):

    total_imgs = 2
    batch_imgs = np.zeros((total_imgs, image_size, image_size, 3))
 
    flipped = np.fliplr(rgb)

    batch_imgs[0,:,:,:] = rgb / 255.0
    batch_imgs[1,:,:,:] = flipped / 255.0

    return batch_imgs

def main(args):
    
    if 'efficientnet' in args.trained_model:
        model = efficientnet.load_model(args.trained_model)
    else:
        model = keras.models.load_model(args.trained_model)

    image_size = model.layers[0].input_shape[1]

    img_paths = glob(args.test_folder + '/*.jpg')
    img_paths.sort()

    # make directory for saving ensemble candidates csv
    if not os.path.exists(args.csv_folder):
        os.mkdir(args.csv_folder)

    total_imgs = len(img_paths)
    print(('total imgs : %d, start : %s, end : %s') % (total_imgs, img_paths[0], img_paths[-1]))

    date = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    csv_file_path = os.path.join(args.csv_folder, 'candidates-' + date + '.csv')

    with open(csv_file_path,'w') as w:
        for i, path in enumerate(img_paths):

            img = keras.preprocessing.image.load_img(path, target_size=(image_size, image_size))
            
            rgb = keras.preprocessing.image.img_to_array(img)

            batch_imgs = make_batch(rgb, image_size)
            predictions = model.predict(batch_imgs)

            prediction = np.mean(predictions, 0)

            file_name = path.split('/')[-1]

            line = file_name + ',' + str(prediction[0]) + ',' + str(prediction[1])
            w.write(line + '\n')

            print('%d/%d : %s processed' % (i+1, total_imgs, file_name))

    print(('%s saved') % (csv_file_path))

        
if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
