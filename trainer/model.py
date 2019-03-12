from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input, MaxPool2D, GlobalAveragePooling2D, Lambda, Conv2D, concatenate, ZeroPadding2D, Layer, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from .utils import get_layers_output_by_name
import random
from .loss import *
from tensorflow.python.lib.io import file_io
import os
import zipfile


def vanila_vgg16():
  vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
  
  first_input = Input(shape=(224,224,3))
  second_input = Input(shape=(224,224,3))

  final_model = tf.keras.models.Model(inputs=[first_input, second_input, vgg_model.input], outputs=vgg_model.output)

  return final_model


def mildnet_without_skip():
  vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
  convnet_output = Dense(1024, activation='relu')(vgg_model.output)
  convnet_output = Dropout(0.6)(convnet_output)
  convnet_output = Dense(1024, activation='relu')(convnet_output)
  convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
  
  first_input = Input(shape=(224,224,3))
  second_input = Input(shape=(224,224,3))

  final_model = tf.keras.models.Model(inputs=[first_input, second_input, vgg_model.input], outputs=convnet_output)

  return final_model


def mildnet_without_skip_big():
  vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
  convnet_output = Dense(2048, activation='relu')(vgg_model.output)
  convnet_output = Dropout(0.6)(convnet_output)
  convnet_output = Dense(2048, activation='relu')(convnet_output)
  convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
  
  first_input = Input(shape=(224,224,3))
  second_input = Input(shape=(224,224,3))

  final_model = tf.keras.models.Model(inputs=[first_input, second_input, vgg_model.input], outputs=convnet_output)

  return final_model


def mildnet_vgg16():
  vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
  
  for layer in vgg_model.layers[:10]:
    layer.trainable = False
  
  intermediate_layer_outputs = get_layers_output_by_name(vgg_model, ["block1_pool", "block2_pool", "block3_pool", "block4_pool"])
  convnet_output = GlobalAveragePooling2D()(vgg_model.output)
  for layer_name, output in intermediate_layer_outputs.items():
    output = GlobalAveragePooling2D()(output)
    convnet_output = concatenate([convnet_output, output])
  
  convnet_output = Dense(2048, activation='relu')(convnet_output)
  convnet_output = Dropout(0.6)(convnet_output)
  convnet_output = Dense(2048, activation='relu')(convnet_output)
  convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
    
  first_input = Input(shape=(224,224,3))
  second_input = Input(shape=(224,224,3))

  final_model = tf.keras.models.Model(inputs=[first_input, second_input, vgg_model.input], outputs=convnet_output)

  return final_model


def mildnet_vgg16_big():
    vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
    
    for layer in vgg_model.layers[:10]:
      layer.trainable = False
    
    intermediate_layer_outputs = get_layers_output_by_name(vgg_model, ["block1_pool", "block2_pool", "block3_pool", "block4_pool"])
    convnet_output = GlobalAveragePooling2D()(vgg_model.output)
    for layer_name, output in intermediate_layer_outputs.items():
      output = GlobalAveragePooling2D()(output)
      convnet_output = concatenate([convnet_output, output])
    
    convnet_output = Dense(2048, activation='relu')(convnet_output)
    convnet_output = Dropout(0.6)(convnet_output)
    convnet_output = Dense(2048, activation='relu')(convnet_output)
    convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
      
    first_input = Input(shape=(224,224,3))
    first_conv = Conv2D(96, kernel_size=(8, 8),strides=(16,16), padding='same')(first_input)
    first_max = MaxPool2D(pool_size=(3,3),strides = (4,4),padding='same')(first_conv)
    first_max = Flatten()(first_max)
    first_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(first_max)

    second_input = Input(shape=(224,224,3))
    second_conv = Conv2D(96, kernel_size=(8, 8),strides=(32,32), padding='same')(second_input)
    second_max = MaxPool2D(pool_size=(7,7),strides = (2,2),padding='same')(second_conv)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(second_max)

    merge_one = concatenate([first_max, second_max])
    merge_two = concatenate([merge_one, convnet_output], axis=1)
    emb = Dense(4096)(merge_two)
    l2_norm_final = Lambda(lambda  x: K.l2_normalize(x,axis=1))(emb)
    final_model = tf.keras.models.Model(inputs=[first_input, second_input, vgg_model.input], outputs=l2_norm_final)

    return final_model


def mildnet_mobilenet():
    vgg_model = MobileNet(weights=None, include_top=False, input_shape=(224,224,3))
    intermediate_layer_outputs = get_layers_output_by_name(vgg_model, ["conv_dw_1_relu", "conv_dw_2_relu", "conv_dw_4_relu", "conv_dw_6_relu", "conv_dw_12_relu"])
    convnet_output = GlobalAveragePooling2D()(vgg_model.output)
    for layer_name, output in intermediate_layer_outputs.items():
      output = GlobalAveragePooling2D()(output)
      convnet_output = concatenate([convnet_output, output])
      
    convnet_output = GlobalAveragePooling2D()(vgg_model.output)
    convnet_output = Dense(1024, activation='relu')(convnet_output)
    convnet_output = Dropout(0.5)(convnet_output)
    convnet_output = Dense(1024, activation='relu')(convnet_output)
    convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
      
    first_input = Input(shape=(224,224,3))
    second_input = Input(shape=(224,224,3))

    final_model = tf.keras.models.Model(inputs=[first_input, second_input, vgg_model.input], outputs=convnet_output)

    return final_model


def mildnet_1024_512():
  vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
  
  for layer in vgg_model.layers[:10]:
    layer.trainable = False
  
  intermediate_layer_outputs = get_layers_output_by_name(vgg_model, ["block1_pool", "block2_pool", "block3_pool", "block4_pool"])
  convnet_output = GlobalAveragePooling2D()(vgg_model.output)
  for layer_name, output in intermediate_layer_outputs.items():
    output = GlobalAveragePooling2D()(output)
    convnet_output = concatenate([convnet_output, output])
  
  convnet_output = Dense(1024, activation='relu')(convnet_output)
  convnet_output = Dropout(0.6)(convnet_output)
  convnet_output = Dense(512, activation='relu')(convnet_output)
  convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
    
  first_input = Input(shape=(224,224,3))
  second_input = Input(shape=(224,224,3))

  final_model = tf.keras.models.Model(inputs=[first_input, second_input, vgg_model.input], outputs=convnet_output)

  return final_model


def mildnet_512_512():
  vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
  
  for layer in vgg_model.layers[:10]:
    layer.trainable = False
  
  intermediate_layer_outputs = get_layers_output_by_name(vgg_model, ["block1_pool", "block2_pool", "block3_pool", "block4_pool"])
  convnet_output = GlobalAveragePooling2D()(vgg_model.output)
  for layer_name, output in intermediate_layer_outputs.items():
    output = GlobalAveragePooling2D()(output)
    convnet_output = concatenate([convnet_output, output])
  
  convnet_output = Dense(512, activation='relu')(convnet_output)
  convnet_output = Dropout(0.6)(convnet_output)
  convnet_output = Dense(512, activation='relu')(convnet_output)
  convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
    
  first_input = Input(shape=(224,224,3))
  second_input = Input(shape=(224,224,3))

  final_model = tf.keras.models.Model(inputs=[first_input, second_input, vgg_model.input], outputs=convnet_output)

  return final_model


def mildnet_512_no_dropout():
  vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
  
  for layer in vgg_model.layers[:10]:
    layer.trainable = False
  
  intermediate_layer_outputs = get_layers_output_by_name(vgg_model, ["block1_pool", "block2_pool", "block3_pool", "block4_pool"])
  convnet_output = GlobalAveragePooling2D()(vgg_model.output)
  for layer_name, output in intermediate_layer_outputs.items():
    output = GlobalAveragePooling2D()(output)
    convnet_output = concatenate([convnet_output, output])
  
  convnet_output = Dense(512, activation='relu')(convnet_output)
  convnet_output = Dense(512, activation='relu')(convnet_output)
  convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
    
  first_input = Input(shape=(224,224,3))
  second_input = Input(shape=(224,224,3))

  final_model = tf.keras.models.Model(inputs=[first_input, second_input, vgg_model.input], outputs=convnet_output)

  return final_model


def mildnet_vgg19():
  vgg_model = VGG19(weights="imagenet", include_top=False, input_shape=(224,224,3))
  
  for layer in vgg_model.layers[:10]:
    layer.trainable = False
  
  intermediate_layer_outputs = get_layers_output_by_name(vgg_model, ["block1_pool", "block2_pool", "block3_pool", "block4_pool"])
  convnet_output = GlobalAveragePooling2D()(vgg_model.output)
  for layer_name, output in intermediate_layer_outputs.items():
    output = GlobalAveragePooling2D()(output)
    convnet_output = concatenate([convnet_output, output])
  
  convnet_output = Dense(2048, activation='relu')(convnet_output)
  convnet_output = Dropout(0.6)(convnet_output)
  convnet_output = Dense(2048, activation='relu')(convnet_output)
  convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
    
  first_input = Input(shape=(224,224,3))
  second_input = Input(shape=(224,224,3))

  final_model = tf.keras.models.Model(inputs=[first_input, second_input, vgg_model.input], outputs=convnet_output)

  return final_model


def mildnet_all_trainable():
  vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
  
  intermediate_layer_outputs = get_layers_output_by_name(vgg_model, ["block1_pool", "block2_pool", "block3_pool", "block4_pool"])
  convnet_output = GlobalAveragePooling2D()(vgg_model.output)
  for layer_name, output in intermediate_layer_outputs.items():
    output = GlobalAveragePooling2D()(output)
    convnet_output = concatenate([convnet_output, output])
  
  convnet_output = Dense(2048, activation='relu')(convnet_output)
  convnet_output = Dropout(0.6)(convnet_output)
  convnet_output = Dense(2048, activation='relu')(convnet_output)
  convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
    
  first_input = Input(shape=(224,224,3))
  second_input = Input(shape=(224,224,3))

  final_model = tf.keras.models.Model(inputs=[first_input, second_input, vgg_model.input], outputs=convnet_output)

  return final_model


def mildnet_vgg16_skip_1():
  vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
  
  for layer in vgg_model.layers[:10]:
    layer.trainable = False
  
  intermediate_layer_outputs = get_layers_output_by_name(vgg_model, ["block2_pool", "block3_pool", "block4_pool"])
  convnet_output = GlobalAveragePooling2D()(vgg_model.output)
  for layer_name, output in intermediate_layer_outputs.items():
    output = GlobalAveragePooling2D()(output)
    convnet_output = concatenate([convnet_output, output])
  
  convnet_output = Dense(2048, activation='relu')(convnet_output)
  convnet_output = Dropout(0.6)(convnet_output)
  convnet_output = Dense(2048, activation='relu')(convnet_output)
  convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
    
  first_input = Input(shape=(224,224,3))
  second_input = Input(shape=(224,224,3))

  final_model = tf.keras.models.Model(inputs=[first_input, second_input, vgg_model.input], outputs=convnet_output)

  return final_model


def mildnet_vgg16_skip_2():
  vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
  
  for layer in vgg_model.layers[:10]:
    layer.trainable = False
  
  intermediate_layer_outputs = get_layers_output_by_name(vgg_model, ["block3_pool", "block4_pool"])
  convnet_output = GlobalAveragePooling2D()(vgg_model.output)
  for layer_name, output in intermediate_layer_outputs.items():
    output = GlobalAveragePooling2D()(output)
    convnet_output = concatenate([convnet_output, output])
  
  convnet_output = Dense(2048, activation='relu')(convnet_output)
  convnet_output = Dropout(0.6)(convnet_output)
  convnet_output = Dense(2048, activation='relu')(convnet_output)
  convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
    
  first_input = Input(shape=(224,224,3))
  second_input = Input(shape=(224,224,3))

  final_model = tf.keras.models.Model(inputs=[first_input, second_input, vgg_model.input], outputs=convnet_output)

  return final_model


def mildnet_vgg16_skip_3():
  vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
  
  for layer in vgg_model.layers[:10]:
    layer.trainable = False
  
  intermediate_layer_outputs = get_layers_output_by_name(vgg_model, ["block4_pool"])
  convnet_output = GlobalAveragePooling2D()(vgg_model.output)
  for layer_name, output in intermediate_layer_outputs.items():
    output = GlobalAveragePooling2D()(output)
    convnet_output = concatenate([convnet_output, output])
  
  convnet_output = Dense(2048, activation='relu')(convnet_output)
  convnet_output = Dropout(0.6)(convnet_output)
  convnet_output = Dense(2048, activation='relu')(convnet_output)
  convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
    
  first_input = Input(shape=(224,224,3))
  second_input = Input(shape=(224,224,3))

  final_model = tf.keras.models.Model(inputs=[first_input, second_input, vgg_model.input], outputs=convnet_output)

  return final_model


def mildnet_vgg16_skip_4():
  vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
  
  for layer in vgg_model.layers[:10]:
    layer.trainable = False
  
  convnet_output = Dense(2048, activation='relu')(vgg_model.output)
  convnet_output = Dropout(0.6)(convnet_output)
  convnet_output = Dense(2048, activation='relu')(convnet_output)
  convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
    
  first_input = Input(shape=(224,224,3))
  second_input = Input(shape=(224,224,3))

  final_model = tf.keras.models.Model(inputs=[first_input, second_input, vgg_model.input], outputs=convnet_output)

  return final_model


def ranknet():
    vgg_model = VGG19(weights="imagenet", include_top=False, input_shape=(224,224,3))
    convnet_output = GlobalAveragePooling2D()(vgg_model.output)
    convnet_output = Dense(4096, activation='relu')(convnet_output)
    convnet_output = Dropout(0.5)(convnet_output)
    convnet_output = Dense(4096, activation='relu')(convnet_output)
    convnet_output = Dropout(0.5)(convnet_output)
    convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
      
    s1_inp = Input(shape=(224,224,3))    
    s1 = MaxPool2D(pool_size=(4,4),strides = (4,4),padding='valid')(s1_inp)
    s1 = ZeroPadding2D(padding=(4, 4), data_format=None)(s1)
    s1 = Conv2D(96, kernel_size=(8, 8),strides=(4,4), padding='valid')(s1)
    s1 = ZeroPadding2D(padding=(2, 2), data_format=None)(s1)
    s1 = MaxPool2D(pool_size=(7,7),strides = (4,4),padding='valid')(s1)
    s1 = Flatten()(s1)

    s2_inp = Input(shape=(224,224,3))    
    s2 = MaxPool2D(pool_size=(8,8),strides = (8,8),padding='valid')(s2_inp)
    s2 = ZeroPadding2D(padding=(4, 4), data_format=None)(s2)
    s2 = Conv2D(96, kernel_size=(8, 8),strides=(4,4), padding='valid')(s2)
    s2 = ZeroPadding2D(padding=(1, 1), data_format=None)(s2)
    s2 = MaxPool2D(pool_size=(3,3),strides = (2,2),padding='valid')(s2)
    s2 = Flatten()(s2)
    
    merge_one = concatenate([s1, s2])
    merge_one_norm = Lambda(lambda  x: K.l2_normalize(x,axis=1))(merge_one)
    merge_two = concatenate([merge_one_norm, convnet_output], axis=1)
    emb = Dense(4096)(merge_two)
    l2_norm_final = Lambda(lambda  x: K.l2_normalize(x,axis=1))(emb)
    
    final_model = tf.keras.models.Model(inputs=[s1_inp, s2_inp, vgg_model.input], outputs=l2_norm_final)

    return final_model


class LRN2D(Layer):
  
  def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):  
    if n % 2 == 0:  
        raise NotImplementedError("LRN2D only works with odd n. n provided: " + str(n)) 
    super(LRN2D, self).__init__(**kwargs) 
    self.alpha = alpha  
    self.k = k  
    self.beta = beta  
    self.n = n  

  def get_output(self, train):  
    X = self.get_input(train) 
    b, ch, r, c = K.shape(X)  
    half_n = self.n // 2  
    input_sqr = K.square(X) 
    extra_channels = K.zeros((b, ch + 2 * half_n, r, c))  
    input_sqr = K.concatenate([extra_channels[:, :half_n, :, :],  
                               input_sqr, 
                               extra_channels[:, half_n + ch:, :, :]],  
                              axis=1) 
    scale = self.k  
    for i in range(self.n): 
        scale += self.alpha * input_sqr[:, i:i + ch, :, :]  
    scale = scale ** self.beta  
    return X / scale  

  def get_config(self): 
    config = {"name": self.__class__.__name__+str(random.randint(1,101)), 
              "alpha": self.alpha,  
              "k": self.k,  
              "beta": self.beta,  
              "n": self.n}  
    base_config = super(LRN2D, self).get_config() 
    return dict(list(base_config.items()) + list(config.items()))


def visnet_lrn2d_model():
    vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
    convnet_output = GlobalAveragePooling2D()(vgg_model.output)
    convnet_output = Dense(4096, activation='relu')(convnet_output)
    convnet_output = Dropout(0.6)(convnet_output)
    convnet_output = Dense(4096, activation='relu')(convnet_output)
    convnet_output = Dropout(0.6)(convnet_output)
    convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
  
    first_input = Input(shape=(224,224,3))
    first_maxpool = MaxPooling2D(pool_size=4, strides=4)(first_input)
    first_conv = Conv2D(96, kernel_size=8, strides=4, activation='relu')(first_maxpool)
    first_lrn2d = LRN2D(n=5)(first_conv)
    first_zero_padding = ZeroPadding2D(padding=(3, 3))(first_lrn2d)
    first_maxpool2 = MaxPooling2D(pool_size=7, strides=4, padding='same')(first_zero_padding)
    first_maxpool2 = Flatten()(first_maxpool2)
    first_maxpool2 = Lambda(lambda  x: K.l2_normalize(x,axis=1))(first_maxpool2)

    second_input = Input(shape=(224,224,3))
    second_maxpool = MaxPooling2D(pool_size=8, strides=8)(second_input)
    second_conv = Conv2D(96, kernel_size=8, strides=4, activation='relu')(second_maxpool)
    second_lrn2d = LRN2D(n=5)(second_conv)
    second_zero_padding = ZeroPadding2D(padding=(1, 1))(second_lrn2d)
    second_maxpool2 = MaxPooling2D(pool_size=3, strides=2, padding='same')(second_zero_padding)
    second_maxpool2 = Flatten()(second_maxpool2)
    second_maxpool2 = Lambda(lambda  x: K.l2_normalize(x,axis=1))(second_maxpool2)

    merge_one = concatenate([first_maxpool2, second_maxpool2])
    merge_two = concatenate([merge_one, convnet_output])
    emb = Dense(4096)(merge_two)
    l2_norm_final = Lambda(lambda  x: K.l2_normalize(x,axis=1))(emb)

    final_model = Model(inputs=[first_input, second_input, vgg_model.input], outputs=l2_norm_final)

    return final_model


def visnet_model():
    vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
    convnet_output = GlobalAveragePooling2D()(vgg_model.output)
    convnet_output = Dense(4096, activation='relu')(convnet_output)
    convnet_output = Dropout(0.6)(convnet_output)
    convnet_output = Dense(4096, activation='relu')(convnet_output)
    convnet_output = Dropout(0.6)(convnet_output)
    convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
      
    first_input = Input(shape=(224,224,3))
    first_conv = Conv2D(96, kernel_size=(8, 8),strides=(16,16), padding='same')(first_input)
    first_max = MaxPool2D(pool_size=(3,3),strides = (4,4),padding='same')(first_conv)
    first_max = Flatten()(first_max)
    first_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(first_max)

    second_input = Input(shape=(224,224,3))
    second_conv = Conv2D(96, kernel_size=(8, 8),strides=(32,32), padding='same')(second_input)
    second_max = MaxPool2D(pool_size=(7,7),strides = (2,2),padding='same')(second_conv)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(second_max)

    merge_one = concatenate([first_max, second_max])
    merge_two = concatenate([merge_one, convnet_output], axis=1)
    emb = Dense(4096)(merge_two)
    l2_norm_final = Lambda(lambda  x: K.l2_normalize(x,axis=1))(emb)
    final_model = tf.keras.models.Model(inputs=[first_input, second_input, vgg_model.input], outputs=l2_norm_final)

    return final_model


def alexnet():
    if not os.path.exists("alexnet_keras.zip"):
        print("Downloading Alexnet Keras Helpers")
        with file_io.FileIO("gs://fynd-open-source/research/MILDNet/alexnet_keras.zip", mode='r') as alexnet_keras:
            with file_io.FileIO("alexnet_keras.zip", mode='w+') as output_f:
                output_f.write(alexnet_keras.read())
        dest_path = "/root/.local/lib/python2.7/site-packages/trainer"
        with zipfile.ZipFile("alexnet_keras.zip", 'r') as zip_ref:
            zip_ref.extractall(dest_path)
            import shutil
            for f in os.listdir("{}/alexnet_keras/".format(dest_path)):
                shutil.copy("{}/alexnet_keras/{}".format(dest_path, f), "{}/{}".format(dest_path, f))
                shutil.copy("{}/alexnet_keras/{}".format(dest_path, f), "{}/{}".format("/user_dir", f))

    os.popen("pip install keras==2.0.4").read()
    from convnets import convnet

    alexnet_model = convnet('alexnet',weights_path="alexnet_weights.h5", heatmap=False)
    convnet_output = GlobalAveragePooling2D()(alexnet_model.get_layer('convpool_5').output)
    convnet_output = Dense(4096, activation='relu')(convnet_output)
    convnet_output = Dropout(0.6)(convnet_output)
    convnet_output = Dense(4096, activation='relu')(convnet_output)
    convnet_output = Dropout(0.6)(convnet_output)
    convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
  
    first_input = Input(shape=(3,227,227))
    first_maxpool = MaxPooling2D(pool_size=4, strides=4)(first_input)
    first_conv = Conv2D(96, kernel_size=8, strides=4, activation='relu')(first_maxpool)
    first_zero_padding = ZeroPadding2D(padding=(3, 3))(first_conv)
    first_maxpool2 = MaxPooling2D(pool_size=7, strides=4, padding='same')(first_zero_padding)
    first_maxpool2 = Flatten()(first_maxpool2)
    first_maxpool2 = Lambda(lambda  x: K.l2_normalize(x,axis=1))(first_maxpool2)

    second_input = Input(shape=(3,227,227))
    second_maxpool = MaxPooling2D(pool_size=8, strides=8)(second_input)
    second_conv = Conv2D(96, kernel_size=8, strides=4, activation='relu')(second_maxpool)
    second_zero_padding = ZeroPadding2D(padding=(1, 1))(second_conv)
    second_maxpool2 = MaxPooling2D(pool_size=3, strides=2, padding='same')(second_zero_padding)
    second_maxpool2 = Flatten()(second_maxpool2)
    second_maxpool2 = Lambda(lambda  x: K.l2_normalize(x,axis=1))(second_maxpool2)

    merge_one = concatenate([first_maxpool2, second_maxpool2])
    merge_two = concatenate([merge_one, convnet_output])
    emb = Dense(4096)(merge_two)
    l2_norm_final = Lambda(lambda  x: K.l2_normalize(x,axis=1))(emb)

    final_model = Model(inputs=[first_input, second_input, alexnet_model.input], outputs=l2_norm_final)

    return final_model