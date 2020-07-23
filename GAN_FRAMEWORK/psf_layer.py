# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Layer  # quitar tensorflow si usa keras solo
import numpy as np
import os
import rawpy
import cv2
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras.constraints import NonNeg
from functions import deta, ifftshift, area_downsampling_tf, compl_exp_tf, transp_fft2d, transp_ifft2d, img_psf_conv,fftshift2d_tf

class Psf_layer(Layer):

    def __init__(self, sensor_size, bands, **kwargs):
        self.sensor_size = sensor_size
        self.bands = bands
        super(Psf_layer, self).__init__(**kwargs)

    def get_config(self):

      config = super().get_config().copy()
      config.update({
          'sensor_size': self.sensor_size,
          'bands': self.bands,
      })
      return config

    def build(self, input_shape):
        ban= np.linspace(1,70,31, dtype=int, endpoint=False)
        self.psfs = np.zeros((self.sensor_size, self.sensor_size, self.bands))        
        psfs_load = loadmat('../psf/spiral_psfs512.mat').get('psf_small')
        for j,i in enumerate(ban):
            rgb = psfs_load[:,:,i]
            rgb = cv2.resize(rgb,(self.sensor_size,self.sensor_size))
            self.psfs[:,:,j] = rgb[:,:]        
        
        '''
        start_index = 208
        name_prefix = './psf/PSF_samuel/IMG_0' #leer las psfs
        self.psfs = np.zeros((self.sensor_size, self.sensor_size, self.bands))
        for i in range(self.bands):
            name_f = name_prefix + str(start_index) + '.CR2'
            raw = rawpy.imread(name_f)
            rgb = raw.postprocess()
            rgb = cv2.resize(rgb,(self.sensor_size,self.sensor_size))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            self.psfs[:,:,i] = rgb[:,:]
            #temp = np.zeros((self.sensor_size, self.sensor_size))
            #temp[123:127,123]=0.1
            #temp[123,123:127]=0.1
            #self.psfs[:,:,i] = temp
            start_index = start_index + 1'''
            
        temp = loadmat('../Sensor_31.mat') #depende del numero de bandas
        self.bgr_response = np.concatenate((temp['B'], temp['G'], temp['R']))
        self.bgr_response = tf.cast(self.bgr_response, dtype=tf.float32)
        self.bgr_response = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.bgr_response, -1), -1), -1)
        self.psfs = tf.convert_to_tensor(self.psfs, dtype=tf.float32)
        self.psfs = tf.expand_dims(self.psfs,axis=0)
        self.psfs = tf.transpose(self.psfs, [1, 2, 0, 3])
        super(Psf_layer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        
        
        output_image = img_psf_conv(inputs, self.psfs)
        output_image = tf.cast(output_image, tf.float32)
        # size bands,with, height, batch
        output_image = tf.transpose(output_image, [3, 1, 2, 0])
        # size 3,bands

        # size 3,bands,1,1

        # size 3,width, height,batch
        # prueba------------------------
        #---------------------------------------------
        output_image = tf.multiply(self.bgr_response, output_image)
        output_image = tf.reduce_sum(output_image, axis=1,keepdims=False)
        # size batch,width, height,3
        output_image = tf.transpose(output_image, [3, 1, 2, 0]) 
        return output_image

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
