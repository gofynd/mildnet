from tensorflow.python.lib.io import file_io
import os
import numpy as np
from tensorflow.keras import backend as K
import zipfile
# from .datagen import ImageDataGeneratorCustom
import logging


def downloads_training_images(data_path, is_cropped=False):
    if not os.path.exists("dataset/tops"):
        if not os.path.exists("dataset"):
            os.makedirs("dataset")

    with file_io.FileIO("gs://ml_shared_bucket/prasanna/datasets/combined/dpf2_s2s_images2.zip", mode='rb') as input_f:
        with file_io.FileIO('dataset/tops.zip', mode='w+') as output_f:
            output_f.write(input_f.read())
    with zipfile.ZipFile("dataset/tops.zip", 'r') as zip_ref:
        zip_ref.extractall("dataset")
    os.rename("dataset/dpf2_s2s_images", "dataset/tops")


def get_train_test_csv(data_path, train_csv, val_csv, is_full_data = False):
    with file_io.FileIO("gs://ml_shared_bucket/prasanna/datasets/combined/train_dpf2_s2s_triplets.csv", mode='rb') as train_f:
        if is_full_data:
            with file_io.FileIO(data_path + train_csv, mode='r') as val_f:
                with file_io.FileIO("dataset/"+train_csv, mode='w+') as output_f:
                    output_f.write(train_f.read()+"\n"+val_f.read())
        else:
            with file_io.FileIO("dataset/"+train_csv, mode='w+') as output_f:
                output_f.write(train_f.read())

    with file_io.FileIO("gs://ml_shared_bucket/prasanna/datasets/combined/val_dpf2_s2s_triplets.csv", mode='rb') as val_f:
        with file_io.FileIO("dataset/"+val_csv, mode='w+') as output_f:
            output_f.write(val_f.read())


def get_layers_output_by_name(model, layer_names):
    return {v: model.get_layer(v).output for v in layer_names}


def backup_file(job_dir, filepath):
    if job_dir.startswith("gs://"):
        with file_io.FileIO(filepath, mode='rb') as input_f:
            with file_io.FileIO(os.path.join(job_dir, filepath), mode='w+') as output_f:
                 output_f.write(input_f.read())


def write_file_and_backup(content, job_dir, filepath):
    with open(filepath, "w") as f:
        f.write(content)
    if job_dir.startswith("gs://"):
        backup_file(job_dir, filepath)


def print_trainable_counts(model):
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    logging.info('Total params: {:,}'.format(trainable_count + non_trainable_count))
    logging.info('Trainable params: {:,}'.format(trainable_count))
    logging.info('Non-trainable params: {:,}'.format(non_trainable_count))

    return trainable_count, non_trainable_count