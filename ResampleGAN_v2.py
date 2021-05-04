#!/usr/bin/env python
# coding: utf-8

# # MAIN JUPYTER NOTEBOOK FOR MINOR PROJECT
# 1. Use tensorflow <b>2.3</b> for prototyping
# 2. Use this link to refer to [docs](https://www.tensorflow.org/api_docs/python/tf/all_symbols) and this to refer to [tutorials](https://www.tensorflow.org/tutorials).
# 3. Our github repo is [here](https://github.com/grajat90/ResampleGAN).
# 
# 
# ## Our network (change as needed):
# 
# <center>
# 
# <img src="https://camo.githubusercontent.com/ebd11f6dea8996adcb01132ccd0526e971f4b12b/68747470733a2f2f696d6775722e636f6d2f4a6a7a555958732e6a7067" width = '60%'/>
# 
# <img src="https://camo.githubusercontent.com/07e22e49a908fe6e243468d335a702854260db56/68747470733a2f2f696d6775722e636f6d2f316973696737432e6a7067" width = '60%' />
# 
# </center>
# 
# #### Changes:
# 1. Use of glorot initializer with ICNR initializer
#     * Removed checkerboard pattern
#     * [paper here](https://arxiv.org/pdf/1707.02937.pdf)
#    
# 2. Used Conv2d networks instead of conv2d transpose
# 3. Used Leaky ReLU instead of Parameterized ReLU in residual networks
# 4. Used tanh activation at the end of generator
# 
# 
# ---
# 
# 

# In[ ]:


import os
from lib.icnr import ICNR
import json
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as img
import tensorflow as tf
import numpy as np
import random
from google.colab import auth, drive
from PIL import Image
import tensorflow_datasets as tfds
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.activations import tanh
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Add, PReLU, LeakyReLU, UpSampling2D, GlobalAveragePooling2D, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.python.keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess


# In[ ]:


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# In[ ]:


from google.colab import auth
auth.authenticate_user()


# In[ ]:


get_ipython().system('gsutil -m cp -r gs://srgan-bucket/dataset ./')
# !mv ./main-gan-data/* ./
# !rm -r -rf main-gan-data


# In[ ]:


(ds_train, ds_validation), info = tfds.load(
    'div2k/bicubic_x4',
    split=['train', 'validation'],
    shuffle_files=True,
    with_info = True,
    try_gcs = True,
    data_dir = './dataset'
)
ds_train=ds_train.prefetch(512)
ds_validation = ds_validation.prefetch(512)


# In[ ]:


tf.compat.v1.RunOptions(
    report_tensor_allocations_upon_oom=True
)


# In[ ]:


def generator(momentum=0.8):
    lr_in = tf.keras.Input(shape=(None,None,3))
    hr_out = Conv2D(filters = 64, kernel_size = (9,9), padding='SAME')(lr_in)  #k9n64s1
    hr_out = B = PReLU(shared_axes=[1, 2])(hr_out)
    for i in range(16):
        B_internal = B
        B_internal = Conv2D(filters = 64, kernel_size = (3,3), padding='SAME')(B_internal) #k3n64s1
        B_internal = BatchNormalization(momentum=momentum)(B_internal)
        B_internal = PReLU(shared_axes=[1, 2])(B_internal)
        B_internal = Conv2D(filters = 64, kernel_size = (3,3), padding='SAME')(B_internal) #k3n64s1
        B_internal = BatchNormalization(momentum=momentum)(B_internal)
        B = Add()([B, B_internal])
    B = Conv2D(filters = 64, kernel_size = (3,3), padding='SAME')(B) #k3n64s1
    B = BatchNormalization(momentum=momentum)(B)
    hr_out = Add()([hr_out, B])
    for i in range(2):
        hr_out = Conv2D(filters = 256, kernel_size = (3,3), padding = "SAME", kernel_initializer=ICNR(GlorotUniform()))(hr_out) #k3n256s1
        hr_out = UpSampling2D(size=2)(hr_out)
        hr_out = LeakyReLU(alpha=0.2)(hr_out)

    hr_out = Conv2D(filters = 3, kernel_size = (9,9), padding = "SAME")(hr_out) #k9n3s1
    hr_out = tanh(hr_out)
    return Model(lr_in, hr_out, name="GAN_GEN")


# In[ ]:


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
    pred = BatchNormalization(momentum=momentum)(pred)
    pred = LeakyReLU()(pred)
    pred = GlobalAveragePooling2D()(pred)


    #with dense

    pred = Dense(1, activation="sigmoid")(pred)

    return Model(img_in, pred, name="GAN_DISC")


# In[ ]:


# with strategy.scope():
gen_model = generator(0.5)
disc_model = discriminator(0.5)
gen_optimizer = Adam(learning_rate=1e-6)
disc_optimizer = Adam(learning_rate=1e-6)
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

# @tf.function
def train_step(lr, hr):
  #with tf.device('/gpu:0'):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        lr = tf.dtypes.saturate_cast(lr, tf.float32)
        hr = tf.dtypes.saturate_cast(hr, tf.float32)
        hr_generated = gen_model(lr, training=True)
        fake_output = disc_model(hr_generated, training=True)
        real_output = disc_model(hr, training = True)
        content_loss = vgg_loss(hr, hr_generated)
        disc_loss = discLoss(real_output, fake_output)
        color_loss = mse(hr,hr_generated)
        gen_loss = content_loss + genLoss(fake_output)*1e-3 + color_loss*0.5

    gen_gradients = gen_tape.gradient(gen_loss, gen_model.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, disc_model.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_gradients, gen_model.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, disc_model.trainable_variables))

    return disc_loss, gen_loss


# In[ ]:


try:
    load_gen = gen_model.load_weights("./checkpoint/GEN")
    load_gen.assert_consumed()
    load_disc = disc_model.load_weights("./checkpoint/DISC")
    load_disc.assert_consumed()
except:
    print("Cannot Load model weight - either running for first time or data/path in not proper format")


# In[ ]:


def saveimg(epoch):
    kid_hr = img.imread("./kid.jpg")
    shp = [kid_hr.shape[0]//4,kid_hr.shape[1]//4]
    kid_lr = tf.image.resize(kid_hr, shp, method="bicubic")
    kid_lr = tf.dtypes.saturate_cast([kid_lr], tf.float32)
    kid_lr = kid_lr/255.0
    hr_gen = gen_model(kid_lr, training=False)[0]
    hr_gen = hr_gen*255.0
    hr_gen = tf.dtypes.saturate_cast(hr_gen, tf.uint8)
    hr_gen = hr_gen.numpy()
    hr_gen = Image.fromarray(hr_gen)
    imgfile = "./it-{}.jpg".format(epoch)
    hr_gen.save(imgfile)


# In[ ]:


data = {}
get_ipython().system('gsutil cp gs://srgan-bucket/checkpoint-gpu/epoch-data-gpu.json ./')
try:
    with open('./epoch-data-gpu.json', 'r') as fp:
    data = json.load(fp)
    data = dict([int(key), value] for key, value in data.items())  
except:
    pass
def epochsave(epoch, g_err, d_err):
    data[int(epoch)] = {'g_err': str(g_err), 'd_err': str(d_err)}
    with open('./epoch-data-gpu.json', 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent="")
        fp.close()
    os.system("gsutil cp ./epoch-data-gpu.json gs://srgan-bucket/checkpoint-gpu/")
    return


# In[ ]:


def rcrop(img, rxseed, ryseed):
    (shapex,shapey) = img.shape[:2]
    # sizex = midx//6
    # sizey = midy//6
    sizex = 280
    sizey = 280
    xstart = random.randint(5,shapex-(sizex+10))
    ystart = random.randint(5,shapey-(sizey+10))
    return (img[xstart:xstart+sizex, ystart:ystart+sizey, :], [sizex, sizey])


# In[ ]:


#batched
possible = [x for x in range(-50,51) if x%4==0]


def train(epochs, batch_size, skip_epochs):
    g_loss = None
    d_loss = None
    try:
    load_gen = gen_model.load_weights("gs://srgan-bucket/checkpoint-gpu/GEN")
    load_disc = disc_model.load_weights("gs://srgan-bucket/checkpoint-gpu/DISC")
    load_gen.assert_consumed()
    load_gen.assert_consumed()
    except:
        print("Cannot Load model weight - either running for first time or data/path in not proper format")
    for epoch in tqdm(range(1,epochs+1)):
        if epoch<=skip_epochs:
            continue
        print(f"========================================\nEpoch - {epoch}")
        gLoss = 0
        dLoss = 0
        lr_batch = []
        hr_batch = []
        rxseed = random.choice(possible)
        ryseed = random.choice(possible)
        for idx, elem in enumerate(tqdm(ds_train)):
            hr = tf.dtypes.saturate_cast(elem['hr'], tf.float32)
            (hr, shape) = rcrop(hr, rxseed, ryseed)
            # lr = tf.cast(elem['lr'], tf.float32)
            # lr = rcrop(lr)
            shape = [x//4 for x in shape if x%4 == 0]
            lr = tf.image.resize(hr, shape, method="bicubic")
            hr = hr/255.0
            lr = lr/255.0
            lr_batch.append(lr)
            hr_batch.append(hr)
            if((idx+1)%batch_size==0):
                d_loss, g_loss = train_step(lr_batch, hr_batch)
                gLoss += g_loss
                dLoss += d_loss
                lr_batch = []
                hr_batch = []
                rxseed = random.choice(possible)
                ryseed = random.choice(possible)
            # d_loss, g_loss = strategy.run(train_step, args=(lr, hr))
            if(not ((idx+1)%batch_size==0)):
                d_loss, g_loss = train_step(lr_batch, hr_batch)
                gLoss += g_loss
                dLoss += d_loss
                lr_batch = []
                hr_batch = []
            print(f"{dLoss/800} - Discriminator Loss \n {gLoss/800} - Generator loss\n========================================\n")
            gen_model.save_weights("gs://srgan-bucket/checkpoint-gpu/GEN")
            disc_model.save_weights("gs://srgan-bucket/checkpoint-gpu/DISC")
            epochsave(epoch, (gLoss/800).numpy(), (dLoss/800).numpy())
            if((epoch)%50==0):
                saveimg(epoch)
try:
    skip_epochs = int(max(k for k,v in data.items()))
except:
    skip_epochs = 0
train(epochs=2500, batch_size=5, skip_epochs=skip_epochs) #593


# In[ ]:


scores = {"nearest":{"psnr":0.0,"ssim":0.0},
          "proposed":{"psnr":0.0,"ssim":0.0}}
bias = [x for x in range(-100,100) if x%4==0]
for elem in tqdm(ds_validation):
    hr = tf.dtypes.saturate_cast(elem['hr'], tf.float32)
    rxseed = random.choice(bias)
    ryseed = random.choice(bias)
    (hr, hr_shape) = rcrop(hr, rxseed, ryseed)
    # hr_shape = list(hr.shape)[:2]
    lr_shape = [x//4 for x in hr_shape if x%4 == 0]
    lr = tf.image.resize(hr, lr_shape, method="bicubic")
    hr = hr/255.0
    lr = lr/255.0
    nearest = tf.image.resize(lr,hr_shape,method="nearest")
    lr = tf.dtypes.saturate_cast([lr], tf.float32)
    proposed = gen_model(lr,training = False)[0]
    scores["nearest"]["psnr"] += tf.image.psnr(hr,nearest,1.0).numpy()/100.0
    scores["nearest"]["ssim"] += tf.image.ssim(hr,nearest,1.0).numpy()/100.0
    scores["proposed"]["psnr"] += tf.image.psnr(hr,proposed,1.0).numpy()/100.0
    scores["proposed"]["ssim"] += tf.image.ssim(hr,proposed,1.0).numpy()/100.0

print(json.dumps(scores, indent=3))


# In[ ]:



load_gen = gen_model.load_weights("./checkpoint/GEN")
# load_gen.assert_consumed()
import matplotlib.image as img
kid_hr = img.imread("./kid.jpg")
shp = [kid_hr.shape[0]//4,kid_hr.shape[1]//4]
kid_lr = tf.image.resize(kid_hr, shp, method="bicubic")
kid_hr = tf.cast(kid_hr, tf.uint8)
kid_lr = tf.cast(kid_lr, tf.uint8)
plt_hr = plt.figure(1)
plt.imshow(kid_hr)
plt_lr = plt.figure(2)
plt.imshow(kid_lr)
kid_lr = tf.cast([kid_lr], tf.float32)
kid_lr = kid_lr/255.0
hr_gen = gen_model(kid_lr, training=False)[0]
hr_gen = hr_gen*255.0
hr_gen = tf.cast(hr_gen, tf.uint8)
plt_fake = plt.figure(3)
plt.imshow(hr_gen)
plt.show()


# In[ ]:


X = []
Y1 = []
Y2 = []
for key, val in data.items():
    X.append(int(key))
    Y1.append(tf.strings.to_number(val["g_err"], tf.float32))
    Y2.append(tf.strings.to_number(val["d_err"], tf.float32))
X = X[1000:]
Y1 = Y1[1000:]
Y2 = Y2[1000:]
plt.figure(1)
plt.plot(X,Y1,'r', label="Generator")
plt.legend()
plt.savefig("GEN-1000.jpg")
plt.figure(2)
plt.plot(X,Y2,'b', label="Discriminator")
plt.legend()
plt.savefig("DISC-1000.jpg")
plt.figure(3)
plt.plot(X,Y1,'r', label="Generator")
plt.plot(X,Y2,'b', label="Discriminator")
plt.legend()
plt.savefig("COMB-1000.jpg")
plt.show()


# In[ ]:




