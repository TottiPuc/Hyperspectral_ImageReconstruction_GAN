# -*- coding: utf-8 -*-
# author: Christian Arcos
from tensorflow.keras.models import load_model
from psf_layer import *
from plot_test import *
from GAN import *
from load_data import *
from functions import *
import shutil
import os


# path configuration

PATH = '../Formated/'
EPOCHS= 5
BATCH=2
print(PATH)


# parameters of the net
BATCH_SIZE = 36; IMG_WIDTH = 500; IMG_HEIGHT = 500; L_bands    = 31; L_imput    = 31

(x_train_lr, x_train_hr),(x_test_lr, x_test_hr) = Build_data_set(IMG_WIDTH=IMG_WIDTH,IMG_HEIGHT=IMG_HEIGHT,
                                          L_bands=L_bands,L_imput=L_imput,BATCH_SIZE=BATCH_SIZE,PATH=PATH)


if os.path.isdir(os.getcwd() +'/output'):
  print('directorio ya existe... removiendo y creando uno nuevo...')
  shutil.rmtree(os.getcwd() + '/output')
  os.makedirs(os.getcwd() + '/output')
else:
  os.makedirs(os.getcwd() + '/output')


print('\n Entrenando modelo ....')
train(x_train_hr,x_train_lr,L_imput,EPOCHS,BATCH)
print('\n Fin del entrenamiento ....')      

print('\n Cargando el modelo para etapa de test .... ')
model = load_model('./output/gen_model5.h5', custom_objects={'smse': smse, 'Psf_layer': Psf_layer })
print('\n Modelo cargado test .... ')
# testando el modelo
path_out = './output'
test_model(model, path_out, x_test_lr, x_test_hr )