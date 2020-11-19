#!/usr/bin/env python
# coding: utf-8

# # MAIN JUPYTER NOTEBOOK FOR MINOR PROJECT
# 1. Use tensorflow <b>2.3</b> for prototyping
# 2. Use this link to refer to [docs](https://www.tensorflow.org/api_docs/python/tf/all_symbols) and this to refer to [tutorials](https://www.tensorflow.org/tutorials).
# 3. Our github repo is [here](https://github.com/grajat90/ResampleGAN).
# 
# 
# ## For quick reference:
# 
# ### Original Networks in SRGAN paper:
# 
# <center>
# 
# <img src="https://miro.medium.com/max/1400/1*T_vCYUgD8UygdMWlgV3ciw.png" width = '60%' />
# 
# </center>
# 
# ### Our network (change as needed):
# 
# <center>
# 
# <img src="https://camo.githubusercontent.com/ebd11f6dea8996adcb01132ccd0526e971f4b12b/68747470733a2f2f696d6775722e636f6d2f4a6a7a555958732e6a7067" width = '60%'/>
# 
# <img src="https://camo.githubusercontent.com/07e22e49a908fe6e243468d335a702854260db56/68747470733a2f2f696d6775722e636f6d2f316973696737432e6a7067" width = '60%' />
# 
# </center>
# 
# ---
# 
# 

# In[ ]:


import logging
import sys
logging.basicConfig(filename='./logs.log',
            filemode='w',
            level=logging.INFO)


# In[1]:





# In[2]:


import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.activations import tanh
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Add, PReLU, LeakyReLU, UpSampling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.python.keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess


# In[3]:


# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
# tf.config.experimental_connect_to_cluster(resolver)
# # This is the TPU initialization code that has to be at the beginning.
# tf.tpu.experimental.initialize_tpu_system(resolver)
# print("All devices: ", tf.config.list_logical_devices('TPU'))


# In[4]:


# strategy = tf.distribute.TPUStrategy(resolver)


# In[ ]:


# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '../google-auth.json'
# storage_client = storage.Client()
# buckets = list(storage_client.list_buckets())
# print(buckets)


# In[8]:


(ds_train, ds_validation), info = tfds.load(
    'div2k/bicubic_x4',
    split=['train', 'validation'],
    shuffle_files=True,
    with_info = True,
    try_gcs = True,
    data_dir = './dataset'
)
print(len(ds_train))
ds_train=ds_train.prefetch(512)
ds_validation = ds_validation.prefetch(512)
#ds_validation=ds_validation.as_numpy_iterator()


# In[9]:


tf.compat.v1.RunOptions(
    report_tensor_allocations_upon_oom=True
)
#tf.config.run_functions_eagerly(False)


# In[10]:


def generator(momentum=0.8):
  lr_in = tf.keras.Input(shape=(None,None,3))
  hr_out = Conv2DTranspose(filters = 64, kernel_size = (9,9), padding='SAME')(lr_in)  #k9n64s1
  hr_out = B = PReLU(shared_axes=[1, 2])(hr_out)
  for i in range(5):
    B_internal = B
    for j in range(2):
      B_internal = Conv2DTranspose(filters = 64, kernel_size = (3,3), padding='SAME')(B_internal) #k3n64s1
      B_internal = BatchNormalization(momentum=momentum)(B_internal)
      B_internal = PReLU(shared_axes=[1, 2])(B_internal)
    B = Add()([B, B_internal])
  B = Conv2DTranspose(filters = 64, kernel_size = (3,3), padding='SAME')(B) #k3n64s1
  B = BatchNormalization(momentum=momentum)(B)
  hr_out = Add()([hr_out, B])
  for i in range(2):
    hr_out = Conv2DTranspose(filters = 256, kernel_size = (3,3), padding = "SAME")(hr_out) #k3n256s1
    hr_out = UpSampling2D(size=2)(hr_out)
    hr_out = PReLU(shared_axes=[1, 2])(hr_out)

  hr_out = Conv2DTranspose(filters = 3, kernel_size = (9,9), padding = "SAME")(hr_out) #k9n3s1
  # hr_out = tanh(hr_out)
  return Model(lr_in, hr_out, name="GAN_GEN")


# In[11]:


def discriminator(momentum=0.8):
  img_in = tf.keras.Input(shape = (None,None,3))
  #k3n64s1
  pred = Conv2D(filters = 64, kernel_size=(3,3), padding = "SAME")(img_in)
  pred = LeakyReLU()(pred)

  #k3n64s2
  pred = Conv2D(filters=64, kernel_size=(3,3), strides=2, padding="VALID")(pred)
  pred = BatchNormalization(momentum=momentum)(pred)
  pred = LeakyReLU()(pred)
  
  #k3n128s1
  pred = Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="SAME")(pred)
  pred = BatchNormalization(momentum=momentum)(pred)
  pred = LeakyReLU()(pred)
  
  #k3n128s2
  pred = Conv2D(filters=128, kernel_size=(3,3), strides=2, padding="VALID")(pred)
  pred = BatchNormalization(momentum=momentum)(pred)
  pred = LeakyReLU()(pred)

  #k3n256s1
  pred = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding="SAME")(pred)
  pred = BatchNormalization(momentum=momentum)(pred)
  pred = LeakyReLU()(pred)

  #k3n256s2
  pred = Conv2D(filters=256, kernel_size=(3,3), strides=2, padding="VALID")(pred)
  pred = BatchNormalization(momentum=momentum)(pred)
  pred = LeakyReLU()(pred)

  #k3n512s1
  pred = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding="SAME")(pred)
  pred = BatchNormalization(momentum=momentum)(pred)
  pred = LeakyReLU()(pred)

  #k3n512s2
  pred = Conv2D(filters=512, kernel_size=(3,3), strides=2, padding="VALID")(pred)
  pred = BatchNormalization(momentum=momentum)(pred)
  pred = LeakyReLU()(pred)

  #avoiding dense layer for dimensional invariance
  pred = Conv2D(filters=1, kernel_size=3, padding="SAME")(pred)

  return Model(img_in, pred, name="GAN_DISC")


# In[12]:


# with strategy.scope():
gen_model = generator(0.5)
disc_model = discriminator(0.5)
gen_optimizer = Adam(learning_rate=1e-4)
disc_optimizer = Adam(learning_rate=1e-4)
bce = tf.keras.losses.BinaryCrossentropy()
mse = tf.keras.losses.MeanSquaredError()
def vgg():
    _ = VGG19(input_shape=(None, None, 3), include_top=False)
    return Model(_.input, _.layers[20].output)
vgg_model = vgg()


def vgg_loss(true_image, fake_image):
  true_image = vgg19_preprocess(true_image)
  fake_image = vgg19_preprocess(fake_image)
  true_features = vgg_model(true_image)
  fake_features = vgg_model(fake_image)
  mseError = mse(true_features, fake_features)
  return mseError


def discLoss(true_output, fake_output):
  disc_fake_loss = bce(tf.zeros_like(fake_output), fake_output)
  disc_true_loss = bce(tf.ones_like(true_output), true_output)
  return disc_fake_loss + disc_true_loss


def genLoss(fake_output):
  gen_loss = bce(tf.ones_like(fake_output), fake_output)
  return gen_loss


def train_step(lr, hr):
  #with tf.device('/gpu:0'):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    lr = tf.cast(lr, tf.float32)
    hr = tf.cast(hr, tf.float32)
    hr_generated = gen_model(lr, training=True)
    fake_output = disc_model(hr_generated, training=True)
    real_output = disc_model(hr, training = True)
    content_loss = vgg_loss(hr, hr_generated)
    disc_loss = discLoss(real_output, fake_output)
    gen_loss = content_loss + genLoss(fake_output)*1e-3

  gen_gradients = gen_tape.gradient(gen_loss, gen_model.trainable_variables)
  disc_gradients = disc_tape.gradient(disc_loss, disc_model.trainable_variables)
  gen_optimizer.apply_gradients(zip(gen_gradients, gen_model.trainable_variables))
  disc_optimizer.apply_gradients(zip(disc_gradients, disc_model.trainable_variables))

  return disc_loss, gen_loss


# In[13]:
mod = os.getenv("model")
chkpt_path = "gs://main-gan-data/chkpt-"+mod
os.mkdir(chkpt_path)

try:
  load_gen = gen_model.load_weights(chkpt_path+"/GEN")
  load_disc = disc_model.load_weights(chkpt_path+"/DISC")
  load_gen.assert_consumed()
  load_gen.assert_consumed()
except:
  logging.info("Cannot Load model weight - either running for first time or data/path in not proper format")


# In[ ]:


def rcrop(img):
  (midx,midy) = img.shape[:2]
  sizex = midx//4
  sizey = midy//4
  midx = midx//2
  midy = midy//2
  xstart = midx - (sizex//2) - 1
  ystart = midy - (sizey//2) - 1
  return img[xstart:xstart+sizex, ystart:ystart+sizey, :]



def train(epochs):
  g_loss = None
  d_loss = None
  for epoch in tqdm(range(epochs)):
    epch = epoch + 1
    log = str("========================================\nEpoch - {}".format(epch))
    logging.info(log)
    gLoss = 0
    dLoss = 0
    for idx, elem in enumerate(tqdm(ds_train)):
      hr = tf.cast(elem['hr'], tf.float32)
      hr = rcrop(hr)
      lr = tf.cast(elem['lr'], tf.float32)
      lr = rcrop(lr)
      hr = hr/255.0
      lr = lr/255.0
      d_loss, g_loss = train_step([lr], [hr])
      gLoss += g_loss
      dLoss += d_loss
      # d_loss, g_loss = strategy.run(train_step, args=(lr, hr))
    log = str("{} - Discriminator Loss \n {} - Generator loss\n========================================\n".format((dLoss/800), (gLoss/800)))
    logging.info(log)
    gen_model.save_weights(chkpt_path+"/GEN")
    disc_model.save_weights(chkpt_path+"/DISC")
  # for epoch in range(epochs):
  #   print("Iter {}/{}, DIV2K BICUBIC 4X".format(epoch, epochs))
  #   for data_next in tqdm(ds_train):
  #     hr = [data_next['hr']/255.0]
  #     lr = [data_next['lr']/255.0]
  #     d_loss, g_loss = train_step(lr, hr)
      
iters = int(os.getenv("iters"))
train(iters)


# In[13]:





# In[21]:





# In[ ]:





# 
