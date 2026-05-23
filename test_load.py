import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'leaf_project.settings')
import django
django.setup()
import tensorflow as tf
from keras.layers import DepthwiseConv2D

class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(**kwargs)

try:
    model = tf.keras.models.load_model(r'C:\Users\prave\Documents\banana_cnn_project\saved_model\banana_disease_model.h5', custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
    print('Loaded successfully!', model is not None)
except Exception as e:
    print('Error:', e)
