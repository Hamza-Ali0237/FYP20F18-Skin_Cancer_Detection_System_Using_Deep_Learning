import tensorflow as tf
import tensorflowjs as tfjs

# load model
model = tf.keras.models.load_model('/Users/macbook/Desktop/20F18_SCDS_FYP/APIs/h10k-model-api/SCDSNet-H10K_Model-1.keras')

tfjs.converters.save_keras_model(model, '/Users/macbook/Desktop/20F18_SCDS_FYP/APIs/h10k-model-api/')