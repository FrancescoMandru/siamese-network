import tensorflow as tf
import keras
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import  MaxPooling2D, Dense, Conv2D, Flatten, Lambda, Dropout, merge


def vggSiameseLeg(input_shape, batch_size):
    """ Base Siamese leg to train
    Input:
      input_shape: input imahe shape
      batch_size: batch size
    Output:
      Siamese Leg model
    """
    input = keras.Input(input_shape, batch_size=batch_size, name='VGGInput')

    vgg_model = keras.applications.VGG16(include_top=False, weights='imagenet', 
                                       pooling='max', input_tensor=input)

    model = Sequential()

    for layer in vgg_model.layers:
      model.add(layer)

    for layer in model.layers:
      layer.trainable = False

    return Model(inputs = [input], outputs = [model.output], name = 'SiameseLeg')

def siameseLeg(input_shape, batch_size):
  """ Base Siamese leg to train
  Input:
    input_shape: input imahe shape
    batch_size: batch size
  Output:
    Siamese Leg model
  """
  input = keras.Input(shape=input_shape, batch_size=batch_size)
  x = Conv2D(8, 
             (5,5),  
             activation='relu',
             kernel_regularizer=tf.keras.regularizers.L2(0.03),
             input_shape=input_shape)(input)
  
  x = MaxPooling2D((2,2))(x)
  x = Conv2D(8, 
             (5,5), 
             activation='relu',
             kernel_regularizer=tf.keras.regularizers.L2(0.03))(x)

  x = MaxPooling2D((2,2))(x)
  x = Conv2D(16, 
             (3,3), 
             activation='relu',
             kernel_regularizer=tf.keras.regularizers.L2(0.03))(x)

  x = MaxPooling2D((2,2))(x)
  x = Flatten()(x)
  out = Dense(256, 
              activation='relu',
              kernel_regularizer=tf.keras.regularizers.L2(0.05))(x)          

  return Model(inputs = [input], outputs = [out], name = 'SiameseLeg')


def siameseNet(siameseLeg, input_shape, output_bias=None, pretrained_leg=False):
  """Full siamese network
  Input: 
    siameseLeg: input siamese leg
    input_shape: input shape of image
  Output:
    Full siamese network
  """
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)

  leg_shape = input_shape[2:]

  left_leg_input = keras.Input(leg_shape, name = 'LeftInput')
  right_leg_input = keras.Input(leg_shape, name = 'RightInput')

  if (pretrained_leg):
    model = vggSiameseLeg(leg_shape,batch_size=input_shape[1])
  else:
    model = siameseLeg(leg_shape, batch_size=input_shape[1])

  left_leg_out = model(left_leg_input)
  right_leg_out = model(right_leg_input)

  shared_layer = (Dense(256, 
                  kernel_regularizer=tf.keras.regularizers.L2(0.03),
                  activation="relu"))
  
  x = shared_layer(left_leg_out)
  y = shared_layer(right_leg_out)

  L1_layer = Lambda(lambda tensor:tf.abs(tensor[0] - tensor[1]))

  # Add the distance function to the network
  merged_model = L1_layer([x, y])

  output = Dense(1, activation='sigmoid',bias_initializer=output_bias)(merged_model)

  return Model([left_leg_input, right_leg_input], outputs=output, name = 'SiameseNetwork')

  L1_layer = Lambda(lambda tensor:tf.abs(tensor[0] - tensor[1]))

  # Add the distance function to the network
  merged_model = L1_layer([left_leg_out, right_leg_out])

  output = Dense(1, activation='sigmoid')(merged_model)

  return Model([left_leg_input, right_leg_input], outputs=output, name = 'SiameseNetwork')