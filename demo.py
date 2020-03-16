import tensorflow as tf
from keras import models as modelo


model = tf.keras.models.load_model('model.h5')

model.summary()