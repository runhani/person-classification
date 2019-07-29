
import os
import matplotlib.pyplot as plt

from keras import applications
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, MaxPooling2D
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

import sys
import argparse
import efficientnet


# Starter Code for Image Classification

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='person classification training code')

    # model name
    parser.add_argument('--model_name', default='xception', type=str, help='', choices=['xception', 'efficientnet'])

    # input
    '''
    db/
        train/
            positive/
                skt_t_p00001.jpg
                skt_t_p00002.jpg
                ...
            negative/
                skt_t_n00001.jpg
                skt_t_n00001.jpg
                ...
        validation/
            positive/
                skt_v_p00001.jpg
                skt_v_p00002.jpg
                ...
            negative/
                skt_v_n00001.jpg
                skt_v_n00002.jpg
                ...

    '''    
    parser.add_argument('--train_data_dir', default='./db/train', type=str, help='root folder path for training (contaning at least two image folders)')
    parser.add_argument('--val_data_dir', default='./db/validation', type=str, help='root folder path for validation (contaning at least two image folders)')

    parser.add_argument('--number_of_classes', default=2, type=int, help='')

    # hyper parameter
    parser.add_argument('--init_lr', default=1e-4, type=float, help='')
    parser.add_argument('--image_size', default=299, type=int, help='')
    parser.add_argument('--train_epoch', default=20, type=int, help='')
    parser.add_argument('--freeze_layer', default=-30, type=int, help='')
    parser.add_argument('--dense_units', default=2048, type=int, help='')
    parser.add_argument('--dropout_rate', default=0.2, type=float, help='')

    # change batch_size according to your GPU memory for speed up
    parser.add_argument('--train_batch_size', default=16, type=int, help='')
    parser.add_argument('--val_batch_size', default=100, type=int, help='')

    return parser.parse_args(argv)

def train(args):
    if 'efficientnet' in args.model_name:
        pretrained_model = efficientnet.EfficientNetB5(weights='imagenet', include_top=False, input_shape=(args.image_size, args.image_size, 3), pooling='avg')
    else:
        pretrained_model = applications.xception.Xception(weights='imagenet', include_top=False, input_shape=(args.image_size, args.image_size, 3), pooling='avg')

    # Freeze the layers except the last N layers
    for layer in pretrained_model.layers[:args.freeze_layer]:
        layer.trainable = False
    
    # Create the model
    model = Sequential()
    
    # Add the transper learning base model
    model.add(pretrained_model)
    
    # Add new layers
    #model.add(Flatten())
    model.add(Dense(args.dense_units, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Dropout(args.dropout_rate))
    model.add(Dense(args.number_of_classes, activation='softmax'))
    
    # Show a summary of the model. Check the number of trainable parameters
    model.summary()    

    # Save the checkpoint with model name
    model_file_path="%s_base.h5" % args.model_name

    # Keep only a single checkpoint, the best over test accuracy.
    checkpoint = ModelCheckpoint(model_file_path,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True,
                                mode='max')


    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Change the batch_size according to your system RAM
    
    train_generator = train_datagen.flow_from_directory(
            args.train_data_dir,
            target_size=(args.image_size, args.image_size),
            batch_size=args.train_batch_size,
            class_mode='categorical')
    
    validation_generator = validation_datagen.flow_from_directory(
            args.val_data_dir,
            target_size=(args.image_size, args.image_size),
            batch_size=args.val_batch_size,
            class_mode='categorical',
            shuffle=False)

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(lr=args.init_lr),
                metrics=['acc'])

    # Train the model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples/train_generator.batch_size ,
        epochs=args.train_epoch,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples/validation_generator.batch_size,
        verbose=1,
        callbacks=[checkpoint])
    
    return history


def view(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Loss')
    plt.legend()
    
    plt.figure()
    
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Accuracy')
    plt.legend()

    
    plt.show()

def count_dirs(folder_path):
    dirs = [o for o in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path,o))]
    return len(dirs)

def check_input_params(args):
    valid = True
    number_of_train_folders = count_dirs(args.train_data_dir)
    number_of_validation_folders = count_dirs(args.val_data_dir)

    if args.number_of_classes != number_of_train_folders:
        print('plz, check [%s] (# of classes:%d) != (# of folders:%d)' % (args.train_data_dir, args.number_of_classes, number_of_train_folders))
        valid = False
    if args.number_of_classes != number_of_validation_folders:
        print('plz, check [%s] (# of classes:%d) != (# of folders:%d)' % (args.val_data_dir, args.number_of_classes, number_of_validation_folders))
        valid = False

    return valid


def main(args):
    if check_input_params(args):
        history = train(args)
        view(history)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
