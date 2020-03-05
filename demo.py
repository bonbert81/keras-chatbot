import tensorflow as tf


model = tf.compat.v1.keras.models.load_model('model.h5')

model.summary()