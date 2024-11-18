# anime_dc_gan

Tugas
<br>
<img src="https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white">
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
<img src="https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=seaborn&logoColor=white">
<img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black">
<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white">
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
<br>
<p style="text-align:center">
    <a href="https://skills.network/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsIBMGPXX0XCEEN72-2022-01-01" target="_blank">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
    </a>
</p>


# **Creating anime characters using Deep Convolutional Generative Adversarial Networks (DCGANs) and Keras**


Estimated time needed: **60** minutes


Imagine if you are in a video game company, your games is famous for its unique characters for every player. With the growth of the player amount, it comes to be a nearly impossible mission to hand plot the characters for millions of players. Your boss plans to keep the unique character creating function in the game, and you need a method to handle the task. <br>

__Generative adversarial networks (GANs) might help!__<br>
It is a class of machine learning frameworks, first published in June 2014 <a href=https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0XCEEN/papers/1406.2661.pdf >[1]</a>. <br>
GANs could generate new data following the statistic features of the data in the training set. GANs is widely used to generate new and realistic photograph that is authentic to human observers. <br>

Convolutional networks (CNNs) has seen huge adoption in computer vision applications. Applying the CNNs to GANs models could help us in building a photo generating model. The combined method is called Deep Convolutional Generative Adversarial Networks (DCGANs). <br>

In this lab, we will first focus on simulated data to better understand GANs. <br> 
Further, we will use the case of massive anime avatar production to introduce how to use DCGANs.<br>
__You will create anime characters like the ones below in this project.__

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0XCEEN/images/face_cartton.png" width="700" alt="Skills Network Logo">


----
<center><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0XCEEN/images/unknown4.jpeg" width="50%"></center>


## __Table of Contents__

<ol>
    <li><a href="#Objectives">Objectives</a></li>
    <li>
        <a href="#Setup">Setup</a>
        <ol>
            <li><a href="#Installing-Required-Libraries">Installing Required Libraries</a></li>
            <li><a href="#Importing-Required-Libraries">Importing Required Libraries</a></li>
            <li><a href="#Defining-Helper-Functions">Defining Helper Functions</a></li>
        </ol>
    </li>
    <li>
        <a href="#Basic:-Generative-Adversarial-Networks-(GANs)">Basic: Generative Adversarial Networks (GANs)</a>
        <ol>
            <li><a href="#Introduction">Introduction</a></li>
            <li><a href="#Toy-Data">Toy Data</a></li>
            <li><a href="#The-Generator">The Generator</a></li>
            <li><a href="#The-Loss-Function-GANs-(Optional)">The Loss Function GANs (Optional)</a></li>
            <li><a href="#Training-GANs">Training GANs</a></li>
        </ol>
    </li>
    <li>
        <a href="#Deep-Convolutional-Generative-Adversarial-Networks-(DCGANs)">Deep Convolutional Generative Adversarial Networks (DCGANs)</a></li>
        <ol>
            <li><a href="#Case-background">Case background</a></li>
            <li><a href="#Loading-the-Dataset">Loading the Dataset</a></li>
            <li><a href="#Creating-Data-Generator">Creating Data Generator</a></li>
            <li><a href="#Generator-and-Discriminator-(for-DCGANs)">Generator and Discriminator  (for DCGANs)</a></li>
            <li><a href="#Defining-Loss-Functions">Defining Loss Functions</a></li>
            <li><a href="#Defining-Optimizers">Defining Optimizers</a></li>
            <li><a href="#Create-Train-Step-Function">Create Train Step Function</a></li>
            <li><a href="#Training-DCGANs">Training DCGANs</a></li>
        </ol>
    <li>
    <a href="#Explore-Latent-Variables">Explore Latent Variables</a>
        <ol>
            <li><a href="#Exercise-1">Exercise 1</a></li>
            <li><a href="#Exercise-2">Exercise 2</a></li>
            <li><a href="#Exercise-3">Exercise 3</a></li>
        </ol>
    </li>
</ol>


## Objectives

After completing this lab, you will be able to:

- __Understand__ the original formulation of GANs, and their two separately trained networks: Generator and Discriminator
- __Implement__ GANs on simulated and real datasets
- __Apply__ DCGANs to a dataset 
- __Understand__ how to train DCGANs 
- __Generate__ an image using a DCGAN
- __Understand__ how changing the input of the latent space of DCGANs changes the generated image 


----


## Setup


For this lab, we will be using the following libraries:

*   [`pandas`](https://pandas.pydata.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0187ENSkillsNetwork31430127-2021-01-01) for managing the data.
*   [`numpy`](https://numpy.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0187ENSkillsNetwork31430127-2021-01-01) for mathematical operations.
*   [`sklearn`](https://scikit-learn.org/stable/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0187ENSkillsNetwork31430127-2021-01-01) for machine learning and machine-learning-pipeline related functions.
*   [`seaborn`](https://seaborn.pydata.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0187ENSkillsNetwork31430127-2021-01-01) for visualizing the data.
*   [`matplotlib`](https://matplotlib.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0187ENSkillsNetwork31430127-2021-01-01) for additional plotting tools.
*   [`keras`](https://keras.io/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0187ENSkillsNetwork31430127-2021-01-01) for loading datasets.
*   [`tensorflow`](https://www.tensorflow.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0187ENSkillsNetwork31430127-2021-01-01) for machine learning and neural network related functions.




### Installing Required Libraries

The following required libraries are pre-installed in the Skills Network Labs environment. However, if you run this notebook commands in a different Jupyter environment (e.g. Watson Studio or Ananconda), you will need to install these libraries by removing the `#` sign before `!mamba` in the code cell below.



```python
# All Libraries required for this lab are listed below. The libraries pre-installed on Skills Network Labs are commented.
# !mamba install -qy pandas==1.3.4 numpy==1.21.4 seaborn==0.9.0 matplotlib==3.5.0 scikit-learn==0.20.1
# Note: If your environment doesn't support "!mamba install", use "!pip install"
```

The following required libraries are __not__ pre-installed in the Skills Network Labs environment. __You will need to run the following cell__ to install them:

_You need ~30 seconds to install._



```python
!mamba install -qy tqdm
```

    Preparing transaction: ...working... done
    Verifying transaction: ...working... done
    Executing transaction: ...working... done


Run the following upgrade and then **RESTART YOUR KERNEL**. Make sure the version of tensorflow imported below is **no less than 2.9.0**. 



```python
!pip3 install  --upgrade tensorflow
```

    Requirement already satisfied: tensorflow in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (1.14.0)
    Collecting tensorflow
      Downloading tensorflow-2.11.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (588.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m588.3/588.3 MB[0m [31m778.2 kB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: absl-py>=1.0.0 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow) (1.4.0)
    Collecting astunparse>=1.6.0 (from tensorflow)
      Downloading astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
    Collecting flatbuffers>=2.0 (from tensorflow)
      Downloading flatbuffers-24.3.25-py2.py3-none-any.whl (26 kB)
    Collecting gast<=0.4.0,>=0.2.1 (from tensorflow)
      Downloading gast-0.4.0-py3-none-any.whl (9.8 kB)
    Requirement already satisfied: google-pasta>=0.1.1 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow) (0.2.0)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow) (1.48.1)
    Collecting h5py>=2.9.0 (from tensorflow)
      Downloading h5py-3.8.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.3/4.3 MB[0m [31m95.6 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hCollecting keras<2.12,>=2.11.0 (from tensorflow)
      Downloading keras-2.11.0-py2.py3-none-any.whl (1.7 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m88.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting libclang>=13.0.0 (from tensorflow)
      Downloading libclang-18.1.1-py2.py3-none-manylinux2010_x86_64.whl (24.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m24.5/24.5 MB[0m [31m59.0 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: numpy>=1.20 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow) (1.21.6)
    Collecting opt-einsum>=2.3.2 (from tensorflow)
      Downloading opt_einsum-3.3.0-py3-none-any.whl (65 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m65.5/65.5 kB[0m [31m13.0 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: packaging in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow) (23.1)
    Collecting protobuf<3.20,>=3.9.2 (from tensorflow)
      Downloading protobuf-3.19.6-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m82.1 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: setuptools in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow) (67.7.2)
    Requirement already satisfied: six>=1.12.0 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow) (1.16.0)
    Collecting tensorboard<2.12,>=2.11 (from tensorflow)
      Downloading tensorboard-2.11.2-py3-none-any.whl (6.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.0/6.0 MB[0m [31m80.7 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hCollecting tensorflow-estimator<2.12,>=2.11.0 (from tensorflow)
      Downloading tensorflow_estimator-2.11.0-py2.py3-none-any.whl (439 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m439.2/439.2 kB[0m [31m50.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: termcolor>=1.1.0 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow) (2.3.0)
    Requirement already satisfied: typing-extensions>=3.6.6 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow) (4.5.0)
    Requirement already satisfied: wrapt>=1.11.0 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow) (1.14.1)
    Collecting tensorflow-io-gcs-filesystem>=0.23.1 (from tensorflow)
      Downloading tensorflow_io_gcs_filesystem-0.34.0-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (2.4 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.4/2.4 MB[0m [31m85.4 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: wheel<1.0,>=0.23.0 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from astunparse>=1.6.0->tensorflow) (0.40.0)
    Collecting google-auth<3,>=1.6.3 (from tensorboard<2.12,>=2.11->tensorflow)
      Downloading google_auth-2.36.0-py2.py3-none-any.whl (209 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m209.5/209.5 kB[0m [31m32.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting google-auth-oauthlib<0.5,>=0.4.1 (from tensorboard<2.12,>=2.11->tensorflow)
      Downloading google_auth_oauthlib-0.4.6-py2.py3-none-any.whl (18 kB)
    Requirement already satisfied: markdown>=2.6.8 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorboard<2.12,>=2.11->tensorflow) (3.4.3)
    Requirement already satisfied: requests<3,>=2.21.0 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorboard<2.12,>=2.11->tensorflow) (2.29.0)
    Collecting tensorboard-data-server<0.7.0,>=0.6.0 (from tensorboard<2.12,>=2.11->tensorflow)
      Downloading tensorboard_data_server-0.6.1-py3-none-manylinux2010_x86_64.whl (4.9 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.9/4.9 MB[0m [31m90.0 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hCollecting tensorboard-plugin-wit>=1.6.0 (from tensorboard<2.12,>=2.11->tensorflow)
      Downloading tensorboard_plugin_wit-1.8.1-py3-none-any.whl (781 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m781.3/781.3 kB[0m [31m77.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: werkzeug>=1.0.1 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorboard<2.12,>=2.11->tensorflow) (2.2.3)
    Collecting cachetools<6.0,>=2.0.0 (from google-auth<3,>=1.6.3->tensorboard<2.12,>=2.11->tensorflow)
      Downloading cachetools-5.5.0-py3-none-any.whl (9.5 kB)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.12,>=2.11->tensorflow) (0.3.0)
    Collecting rsa<5,>=3.1.4 (from google-auth<3,>=1.6.3->tensorboard<2.12,>=2.11->tensorflow)
      Downloading rsa-4.9-py3-none-any.whl (34 kB)
    Collecting requests-oauthlib>=0.7.0 (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.12,>=2.11->tensorflow)
      Downloading requests_oauthlib-2.0.0-py2.py3-none-any.whl (24 kB)
    Requirement already satisfied: importlib-metadata>=4.4 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from markdown>=2.6.8->tensorboard<2.12,>=2.11->tensorflow) (4.11.4)
    Requirement already satisfied: charset-normalizer<4,>=2 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<2.12,>=2.11->tensorflow) (3.1.0)
    Requirement already satisfied: idna<4,>=2.5 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<2.12,>=2.11->tensorflow) (3.4)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<2.12,>=2.11->tensorflow) (1.26.15)
    Requirement already satisfied: certifi>=2017.4.17 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<2.12,>=2.11->tensorflow) (2023.5.7)
    Requirement already satisfied: MarkupSafe>=2.1.1 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from werkzeug>=1.0.1->tensorboard<2.12,>=2.11->tensorflow) (2.1.1)
    Requirement already satisfied: zipp>=0.5 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.12,>=2.11->tensorflow) (3.15.0)
    Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.12,>=2.11->tensorflow) (0.5.0)
    Collecting oauthlib>=3.0.0 (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.12,>=2.11->tensorflow)
      Downloading oauthlib-3.2.2-py3-none-any.whl (151 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m151.7/151.7 kB[0m [31m25.2 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: tensorboard-plugin-wit, libclang, flatbuffers, tensorflow-io-gcs-filesystem, tensorflow-estimator, tensorboard-data-server, rsa, protobuf, opt-einsum, oauthlib, keras, h5py, gast, cachetools, astunparse, requests-oauthlib, google-auth, google-auth-oauthlib, tensorboard, tensorflow
      Attempting uninstall: tensorflow-estimator
        Found existing installation: tensorflow-estimator 1.14.0
        Uninstalling tensorflow-estimator-1.14.0:
          Successfully uninstalled tensorflow-estimator-1.14.0
      Attempting uninstall: protobuf
        Found existing installation: protobuf 4.21.8
        Uninstalling protobuf-4.21.8:
          Successfully uninstalled protobuf-4.21.8
      Attempting uninstall: keras
        Found existing installation: Keras 2.1.6
        Uninstalling Keras-2.1.6:
          Successfully uninstalled Keras-2.1.6
      Attempting uninstall: h5py
        Found existing installation: h5py 2.8.0
        Uninstalling h5py-2.8.0:
          Successfully uninstalled h5py-2.8.0
      Attempting uninstall: gast
        Found existing installation: gast 0.5.3
        Uninstalling gast-0.5.3:
          Successfully uninstalled gast-0.5.3
      Attempting uninstall: tensorboard
        Found existing installation: tensorboard 1.14.0
        Uninstalling tensorboard-1.14.0:
          Successfully uninstalled tensorboard-1.14.0
      Attempting uninstall: tensorflow
        Found existing installation: tensorflow 1.14.0
        Uninstalling tensorflow-1.14.0:
          Successfully uninstalled tensorflow-1.14.0
    Successfully installed astunparse-1.6.3 cachetools-5.5.0 flatbuffers-24.3.25 gast-0.4.0 google-auth-2.36.0 google-auth-oauthlib-0.4.6 h5py-3.8.0 keras-2.11.0 libclang-18.1.1 oauthlib-3.2.2 opt-einsum-3.3.0 protobuf-3.19.6 requests-oauthlib-2.0.0 rsa-4.9 tensorboard-2.11.2 tensorboard-data-server-0.6.1 tensorboard-plugin-wit-1.8.1 tensorflow-2.11.0 tensorflow-estimator-2.11.0 tensorflow-io-gcs-filesystem-0.34.0


### Importing Required Libraries

_We recommend you import all required libraries in one place (here):_

_You need ~1 minute to import._



```python
import warnings
warnings.simplefilter('ignore')
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
print(f"tensorflow version: {tf.__version__}")
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Conv2DTranspose,BatchNormalization,ReLU,Conv2D,LeakyReLU
import time

import keras

from IPython import display
import skillsnetwork
print(f"skillsnetwork version: {skillsnetwork.__version__}")

import matplotlib.pyplot as plt
%matplotlib inline

import os
from os import listdir
from pathlib import Path
import imghdr

from tqdm import tqdm
```

    2024-11-11 17:52:34.231058: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F AVX512_VNNI FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-11-11 17:52:34.431033: I tensorflow/core/util/port.cc:104] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-11-11 17:52:34.436175: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2024-11-11 17:52:34.436211: I tensorflow/compiler/xla/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    2024-11-11 17:52:35.196787: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory
    2024-11-11 17:52:35.196931: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory
    2024-11-11 17:52:35.196944: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.


    tensorflow version: 2.11.0
    skillsnetwork version: 0.20.6


### Defining Helper Functions



```python
# This function will allow us to easily plot data taking in x values, y values, and a title
def plot_distribution(real_data,generated_data,discriminator=None,density=True):
    
    plt.hist(real_data.numpy(), 100, density=density, facecolor='g', alpha=0.75, label='real data')
    plt.hist(generated_data.numpy(), 100, density=density, facecolor='r', alpha=0.75,label='generated data q(z) ')
    
    if discriminator:
        max_=np.max([int(real_data.numpy().max()),int(generated_data.numpy().max())])
        min_=np.min([int(real_data.numpy().min()),int(generated_data.numpy().min())])
        x=np.linspace(min_, max_, 1000).reshape(-1,1)
        plt.plot(x,tf.math.sigmoid(discriminator(x,training=False).numpy()),label='discriminator',color='k')
        plt.plot(x,0.5*np.ones(x.shape),label='0.5',color='b')
        plt.xlabel('x')
        
    plt.legend()
    plt.show()

def plot_array(X,title=""):
    
    plt.rcParams['figure.figsize'] = (20,20) 

    for i,x in enumerate(X[0:5]):
        x=x.numpy()
        max_=x.max()
        min_=x.min()
        xnew=np.uint(255*(x-min_)/(max_-min_))
        plt.subplot(1,5,i+1)
        plt.imshow(xnew)
        plt.axis("off")

    plt.show()
```

----


## Basic: Generative Adversarial Networks (GANs)


### Introduction


__Generative Adversarial Networks (GANs)__ are generative models that convert random samples of one distribution into another distribution. They have several applications, like the following:

*  Generate Examples for Image Datasets
*  Generate Photographs of Human Faces
*  Generate Realistic Photographs
*  Generate Cartoon Characters
*  Image-to-Image Translation
*  Text-to-Image Translation
*  Face Frontal View Generation
*  Generate New Human Poses
*  Face Aging
*  Photo Blending
*  Super Resolution
*  Photo Inpainting
*  Clothing Translation
*  Video Prediction

In this GANs section of the Lab, we will use a toy example to help understand the __basic theoretical principles__ behind GANs. The original form of GANs consisted of a __discriminator__ and a __generator__; let's use the analogy of a currency forger and the police. 

The Generator is the currency forger, and the output is the counterfeit, for example, a 100-dollar bill. The discriminator is analogous to the police taking the counterfeit and trying to determine if it's real by comparing it to a real $100 bill. In real life, if the counterfeit is easy to detect, the forger will adapt; conversely, the police will also improve; GANs emulate this game of cat and mouse.  

<center><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0XCEEN/images/General%20diagram.png" alt="generator image" width="700px"></center>

What makes GANs interesting is that the __discriminator and generator continuously improve__ each other by a well-formulated cost function that backpropagates the errors. GANs are a family of algorithms that use _learning by comparison_. In the lab, we will review the original formulation and use a simulated dataset. We will also point you to some more advanced methods and issues you will encounter with the real datasets for the next lab. 


### Toy Data

Consider the following data, $\mathbf{x}$, that is normally distributed $\mathbf{x} \sim \mathcal{N}(\mathbf{x}|10,1) $ with a mean of 10 and a standard deviation of 1. Now we would like to randomly sample data from this distribution.



```python
mean = [10]
cov = [[1]]
X = tf.random.normal((5000,1),mean=10,stddev=1.0)

print("mean:",np.mean(X))
print("standard deviation:",np.std(X))
```

    mean: 9.990422
    standard deviation: 1.0177704


    2024-11-11 17:52:46.843365: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
    2024-11-11 17:52:46.843419: W tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:265] failed call to cuInit: UNKNOWN ERROR (303)
    2024-11-11 17:52:46.843453: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (jupyterlab-marshallalka): /proc/driver/nvidia/version does not exist
    2024-11-11 17:52:46.843901: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F AVX512_VNNI FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


We also have the data sample, z, which is also normally distributed $\mathbf{z} \sim \mathcal{N}(\mathbf{z}|0,2) $, with mean of 0 and a standard deviation of 2:




```python
Z = tf.random.normal((5000,1),mean=0,stddev=2)
```


```python
print("mean:",np.mean(Z))
print("standard deviation:",np.std(Z))
```

    mean: -0.009200496
    standard deviation: 1.9967928


Let's compare the two distributions:



```python
plot_distribution(X,Z,discriminator=None,density=True)
```


    
![png](output_30_0.png)
    


Let's create our first generative model by adding 10 to every sample of $z$. We will call the result $\hat{\mathbf{x}}$  as it's an approximation of $\mathbf{x}$. It is not too difficult to show that $\hat{\mathbf{x}} \sim \mathcal{N}(\mathbf{x}|10,1)$.Xhat=Z+10



```python
Xhat=Z+10
```

We see that the mean and standard deviation are almost identical 



```python
print("mean:",np.mean(Xhat))
print("standard deviation:",np.std(Xhat))
```

    mean: 9.9908
    standard deviation: 1.9967928


Similarly for the histograms 



```python
plot_distribution(X,Xhat,discriminator=None,density=True)
```


    
![png](output_36_0.png)
    


In the case above, since we just add 10 to the latent variable $z$, we transform $z$ using a deterministic function. We can call this an implicit generative model.


### The Generator


<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0XCEEN/images/Unknown.png" width="300px">

There are two networks involved in a GAN, the Generator and the Discriminator. Let's understand the Generator network first.

The Generator is a neural network denoted by $G$; the idea is that a neural network can approximate any function (by the [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsIBMGPXX0XCEEN72-2022-01-01)), so you should be able to generate data samples from any type of distribution. 

Our goal is to convert the samples, $\mathbf{z}$, to one that approximates $\hat{\mathbf{x}}$,  i.e $\hat{\mathbf{x}}=G(\mathbf{z})$. Let's build a simple Generator $G(\mathbf{z})=\mathbf{W}^{T}\mathbf{z}+\mathbf{b} $ using Keras.

The following is a function that outputs a generator using Kera's Sequential model object.



```python
def make_generator_model():
    generator = tf.keras.Sequential()
    generator.add(layers.Dense(1))
    return generator
```

We can use the Generator to convert $\mathbf{z}$ and make a prediction $\hat{\mathbf{x}}$, and display the histogram of the distributions of $\hat{\mathbf{x}}$ and $\mathbf{x}$. As the model is not trained, the trained distributions are quite different:



```python
generator=make_generator_model()

Xhat = generator(Z, training=False)
plot_distribution(real_data=X,generated_data=Xhat)
```


    
![png](output_42_0.png)
    


We will discuss the use of the parameter ```training=False``` later on.


### The Discriminator 


<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0XCEEN/images/Unknown1.png" width="300px">

The discriminator $D(\mathbf{x})$ is a neural network that learns to distinguish between actual and generated samples. The simplest Discriminator is a simple logistic regression function. Let's create a discriminator in Keras with one Dense layer; we leave the logistic function out as it will be incorporated in the cost function, which is the convention in Keras.



```python
def make_discriminator_model():
    discriminator=tf.keras.Sequential()
    discriminator.add(layers.Dense(1))
    return discriminator

discriminator=make_discriminator_model()
```

The discriminator and generator are randomly initialized, but we can plot the output of each and compare it to the true data distribution, with the generated data in red and the real data in green, and the logistic function as a function of the x axis. We also include the threshold. If the output of the logistic function is less than 0.5, the sample is classified as generated data; conversely, if the output is greater than 0.5, the sample will be classified as data that came from the real distribution. 



```python
plot_distribution(real_data=X,generated_data=Xhat,discriminator=discriminator)
```


    
![png](output_48_0.png)
    


Applying the sigmoid function to the discriminator output, we get the probabilites that the samples belong to the real distribution. We can count the number of true samples that the discriminator correctly classifies. 

For the real data, the discriminator successfully assigns a probability greater than 0.5 for all 5000 samples:



```python
py_x=tf.math.sigmoid(discriminator(X,training=False))
np.sum(py_x>0.5)
```




    5000



For the generated data, only a part of the 5000 samples are classified as having more than 50% chance of coming from the real distribution.



```python
py_x=discriminator(Xhat)
np.sum(py_x>0.5)
```




    1886



We can also use the following to find the average value of the sigmoid function for all the samples.  



```python
def get_accuracy(X,Xhat):
    total=0
    py_x=tf.math.sigmoid(discriminator(X,training=False))
    total=np.mean(py_x)
    py_x=tf.math.sigmoid(discriminator(Xhat,training=False))
    total+=np.mean(py_x)
    return total/2
```


```python
get_accuracy(X,Xhat)
```




    0.7500371932983398



In many cases, we can instead study the difference in the distribution; in this case, the discriminator is called a <a href='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0XCEEN/papers/2107.06700.pdf'>Critic</a>, a real-valued function.


### The Loss Function GANs (Optional) 
GANs convert an unsupervised learning problem to a supervised one. Instead of formulating the problem like a two-player minimax game with a value function like in <a href=https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0XCEEN/papers/1406.2661.pdf >[1]</a>, we can treat the problem of maximizing the familiar log-likelihood of the logistic function analogous to minimizing the cross-entropy loss, then incorporate the generator and discriminator.

___Discriminator___

In order to train the GANS, we start off with standard maximization of the likelihood for the discriminator for the standard dataset $\mathcal{D}=\{{(x_1, y_1), ..., (x_N, y_N)}\}$:

$$V(D)=\sum_{n=1}^N \left( y_n \ln(D(\mathbf{x}_n))+(1-y_n) \ln(1-D(\mathbf{x}_n))\right)$$

Where $y=1$ for samples from the true distribution and $y=0$ for samples from the generator. The goal is to maximize this term with respect to $D$:

$$max_{D}(V(D))$$


To also incorporate the generated samples, we augment the right side of the equation with the generated $k$th sample $\hat{\mathbf{x}}_k$. As they are not part of the dataset $k \notin \mathcal{D} $, we have to include a second summation where $y=0$. Finally, combining the cases of $y=1$ and $y=0$, we get:

$$V(D)=\sum_{ n	\in \mathcal{D}}  \ln(D(\mathbf{x}_n))+\sum_{k 	\notin \mathcal{D}} \ln(1-D(\hat{\mathbf{x}}_k) ) $$


___Generator___ 

For the generator we simply replace $\hat{\mathbf{x}}_k$ with the $G(\mathbf{z}_k)$ . 


$$V(G,D)=\sum_{n	\in \mathcal{D}} \ln(D(\mathbf{x}_n))+\sum_{k 	\notin \mathcal{D}} \ln(1-D(G(\mathbf{z}_k))) $$

As this is a density estimation problem, it is common to replace the summation with the expected value like in <a href=https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0XCEEN/papers/1406.2661.pdf >[1]</a>. We replace the summations with an expectation where $p(\mathbf{x})$ is the true distribution and $p(\mathbf{z})$ is the distribution of $\mathbf{z}$.


$$V(D,G)=\mathbb{E}_{x\sim p(\mathbf{x})} \ln(D(\mathbf{x})) + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} \ln(1-D(G(\mathbf{z}) )) $$

As we are trying to trick the discriminator, we would like to find a $G$ that minimize the above expression, such as:

$$min_{G} max_{D} V(D,G)$$


### Training GANs 

GANs are quite difficult to train, even for a simple example. Let's start off with training the generator in practice. 

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0XCEEN/images/unknown3.jpeg" width="500px">

___Training Generator___

$log(1 − D(G(\mathbf{z})))$ is difficult to work with as $D(G(\mathbf{z}))$ is near one or zero for the first few iterations. This is because the generator is not yet properly trained, and the discriminator can easily distinguish between the generated and actual samples. Therefore we maximize $log(D(G(\mathbf{z}_k)) )$.
 
Although the output of the generator passes through the discriminator, we do not update the generator in the optimization step, hence we set the parameter ```training=False``` in the actual training steps.


Instead of maximizing the term, we can take the negative and minimize it. The resultant expression can be calculated in Keras using the cross-entropy loss where all the target values are set to one:

$$\sum_{k 	\notin \mathcal{D}} log(1 - D(G(\mathbf{z}_k)) )$$



```python
# This method returns a helper function to compute crossentropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def generator_loss(Xhat):
    return cross_entropy(tf.ones_like(Xhat), Xhat)
```

___Training Discriminator___

We can also use the cross-entropy to train the discriminator; we simply multiply $V(G,D)$ by a negative number, set $y=0$ for the generated values and $y=1$ for the real values. We do not update the generator parameters. 

$$V(G)=\sum_{n	\in \mathcal{D}} (\ln(D(\mathbf{x}_n)))+\sum_{k 	\notin \mathcal{D}} \ln(1-D(G(\mathbf{z}_k) )) $$


The first term is the real loss and the second is the fake loss in Keras.



```python
def discriminator_loss(X, Xhat):
    real_loss = cross_entropy(tf.ones_like(X), X)
    fake_loss = cross_entropy(tf.zeros_like(Xhat), Xhat)
    total_loss = 0.5*(real_loss + fake_loss)
    return total_loss
```

We create the optimizer for the discriminator and generator:



```python
generator_optimizer = tf.keras.optimizers.Adam(5e-1,beta_1=0.5,beta_2=0.8)

discriminator_optimizer = tf.keras.optimizers.Adam(5e-1,beta_1=0.5, beta_2=0.8)
```

We now train the model; as the dataset is small, we will use batch gradient descent. 

For each iteration we will generate $M$ real examples $\{\mathbf{x}_{1}, ...,\mathbf{x}_{M}\}$, these are from the generating distribution $p(\mathbf{x})$. This would be our actual dataset if we used real data.  

We will then generate a sample batch of $M$ noise samples $\{\mathbf{z}_{1}, ...,\mathbf{z}_{M}\}$ from noise prior $p(\mathbf{z})$ and convert the result to a generated image using the generator $\{\hat{\mathbf{x}}_{1}, ...,\hat{\mathbf{x}}_{M}\}$. 

We determine the output of the discriminator for both the real and generated samples. We calculate the loss and then update the discriminator and generator through their respective stochastic gradients.


The convergence of GAN training is a subject in itself. But let's explore a method that works for this simple dataset. Intuitively, we know that if our generated data is identical to our actual data, the probability of correctly classifying is random. Therefore if the generated and actual data are of equal proportion, $D(\mathbf{x}_n)=0.5$ and $D(\hat{\mathbf{x}}_n)=0.5$.  

We only display iterations where the average discriminator output gets closer to 50% for both the generated data and actual data.



```python
#parameters for training 
epochs=20
BATCH_SIZE=5000
noise_dim=1
epsilon=100 


#discrimator and gernerator 
tf.random.set_seed(0)
discriminator=make_discriminator_model()
generator=make_generator_model()

tf.config.run_functions_eagerly(True)



gen_loss_epoch=[]
disc_loss_epoch=[]
plot_distribution(real_data=X,generated_data=Xhat,discriminator=discriminator )
print("epoch",0)

for epoch in tqdm(range(epochs)):
    #data for the true distribution of your real data samples training ste
    x = tf.random.normal((BATCH_SIZE,1),mean=10,stddev=1.0)
    #random samples it was found if you increase the standard deviation, you get better results 
    z= tf.random.normal([BATCH_SIZE, noise_dim],mean=0,stddev=10)
    # needed to compute the gradients for a list of variables.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        #generated sample 
        xhat = generator(z, training=True)
        #the output of the discriminator for real data 
        real_output = discriminator(x, training=True)
        #the output of the discriminator  data
        fake_output = discriminator(xhat, training=True)
        #loss for each 
        gen_loss= generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    # Compute the gradients for gen_loss and generator
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    # Compute the gradients for gen_loss and discriminator
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # Ask the optimizer to apply the processed gradients
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
  
  # Save and display the generator and discriminator if the performance increases 
    if abs(0.5-get_accuracy(x,xhat))<epsilon:
        epsilon=abs(0.5-get_accuracy(x,xhat))
        generator.save('generator')
        discriminator.save('discriminator')
        print(get_accuracy(x,xhat))
        plot_distribution(real_data=X,generated_data=xhat,discriminator=discriminator )
        print("epoch",epoch)
```


    
![png](output_69_0.png)
    


      0%|          | 0/20 [00:00<?, ?it/s]

    epoch 0
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    INFO:tensorflow:Assets written to: generator/assets
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    INFO:tensorflow:Assets written to: discriminator/assets
    0.7256608009338379



    
![png](output_69_3.png)
    


      5%|▌         | 1/20 [00:03<01:01,  3.24s/it]

    epoch 0
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    INFO:tensorflow:Assets written to: generator/assets
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    INFO:tensorflow:Assets written to: discriminator/assets
    0.6659221053123474



    
![png](output_69_6.png)
    


     10%|█         | 2/20 [00:05<00:49,  2.77s/it]

    epoch 1
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    INFO:tensorflow:Assets written to: generator/assets
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    INFO:tensorflow:Assets written to: discriminator/assets
    0.3915441036224365



    
![png](output_69_9.png)
    


     15%|█▌        | 3/20 [00:08<00:46,  2.73s/it]

    epoch 2
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    INFO:tensorflow:Assets written to: generator/assets
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    INFO:tensorflow:Assets written to: discriminator/assets
    0.567802369594574



    
![png](output_69_12.png)
    


     35%|███▌      | 7/20 [00:10<00:15,  1.22s/it]

    epoch 6
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    INFO:tensorflow:Assets written to: generator/assets
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    INFO:tensorflow:Assets written to: discriminator/assets
    0.4632177948951721



    
![png](output_69_15.png)
    


     60%|██████    | 12/20 [00:13<00:05,  1.42it/s]

    epoch 7


    100%|██████████| 20/20 [00:13<00:00,  1.44it/s]


For more on training GANs check out the following <a href="https://jonathan-hui.medium.com/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsIBMGPXX0XCEEN72-2022-01-01">blog</a>. We can display the best performing model



```python
generator=make_generator_model()
generator= models.load_model('generator')
xhat=generator(z)
discriminator=models.load_model('discriminator')
plot_distribution(real_data=X,generated_data=xhat,discriminator=discriminator )
```

    WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.
    WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.



    
![png](output_71_1.png)
    


----


In the content above, you learned about the working mechanics of Generative Adversarial Networks (GANs) and their various applications, such as Image Generation. However, GANs have also been known to be unstable to train, and often, the generated images suffer from being noisy and incomprehensible.

For a improved result in the case example, we are applying Convolutional Neural Networks to GANS. They are called Deep Convolutional Generative Adversarial Networks (DCGANs). 
We will build and train DCGANs in the following content, using several approaches introduced in the original <a href="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0XCEEN/papers/1511.06434.pdf">DCGANs paper</a>. 



----


## Deep Convolutional Generative Adversarial Networks (DCGANs)


### Case background


In the case example, you work for an online anime video game company; the company would like to create a unique anime avatar for a game for each player. As there are millions of players, you must use a DCGANs to create each character.

The proposed approaches are summarized here:

- Replace any pooling layers with **strided convolutions (discriminator)** and **fractional-strided
convolutions (generator)**.
- Use **batchnorm** in both the generator and the discriminator.
- **Remove fully connected hidden layers** for deeper architectures.
- Use **ReLU** activation in generator for all layers except for the output, which uses **Tanh**.
- Use **LeakyReLU** activation in the discriminator for all layers except for the output, which uses **Sigmoid**.
- Use **Adam optimizer**.  

These approaches will result in more stable training of deeper generative models.


### Loading the Dataset

We will mainly work with the Anime Face dataset from [Kaggle](https://www.kaggle.com/datasets/splcher/animefacedataset?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsIBMGPXX0XCEEN72-2022-01-01). The original dataset has 63,632 "high-quality" anime faces, but to make the models train faster in this lab, we randomly sampled 20,000 images and prepared a dataset called `cartoon_20000`. 

Let's download the smaller dataset using the Skills Network library's `prepare` function:



```python
dataset_url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module6/cartoon_20000.zip"
await skillsnetwork.prepare(dataset_url, overwrite=True)
```


    Downloading cartoon_20000.zip:   0%|          | 0/131046164 [00:00<?, ?it/s]



      0%|          | 0/20001 [00:00<?, ?it/s]


    Saved to '.'


The Anime Face or the Cartoon images are stored in the `cartoon_2000` folder in your current working directory. As a preprocessing step, we have removed any files that are not proper image formats (based on the file extensions) and any duplicate images.


### Creating Data Generator


First, we declare some properties of our images, including image height, image width, and batch size.



```python
img_height, img_width, batch_size=64,64,128
```

Next, we create a Keras <code>image_dataset_from_directory</code> object with a specified image directory and the parameters are defined as above. This process may take some time:



```python
train_ds = tf.keras.utils.image_dataset_from_directory(directory='cartoon_20000',
                                                       image_size=(img_height, img_width),
                                                       batch_size=batch_size,
                                                       label_mode=None)
```

    Found 20000 files belonging to 1 classes.


The `train_ds` we defined is a `tf.data.Dataset` that yields batches of images with `image_size = (64, 64)` from the directory specified or subdirectories (if any).


**(OPTIONAL)** If you are running this notebook locally and you have multiple cores, then we can use the runtime to tune the value dynamically at runtime as follows:



```python
#AUTOTUNE = tf.data.experimental.AUTOTUNE

#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
```

We apply the Lambda function on `train_ds` to normalize the pixel values of all the input images from $[0, 255]$ to $[-1, 1]$:



```python
normalization_layer = layers.experimental.preprocessing.Rescaling(scale= 1./127.5, offset=-1)
normalized_ds = train_ds.map(lambda x: normalization_layer(x))
```

    WARNING:tensorflow:From /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages/tensorflow/python/autograph/pyct/static_analysis/liveness.py:83: Analyzer.lamba_check (from tensorflow.python.autograph.pyct.static_analysis.liveness) is deprecated and will be removed after 2023-09-23.
    Instructions for updating:
    Lambda fuctions will be no more assumed to be used in the statement where they are used, or at least in the same block. https://github.com/tensorflow/tensorflow/issues/56089


Let's take one batch of images for displaying:



```python
images=train_ds.take(1)
```

Convert the batch dimension to the indexes in a list:



```python
X=[x for x in images]
```

We can then plot the first five images in the batch using the function   ```plot_array```:



```python
plot_array(X[0])
```


    
![png](output_96_0.png)
    


###  Generator and Discriminator (for DCGANs)


___Building the Generator___


The Generator is comprised of several layers of transposed convolution, the opposite of convolution operations.

- Each Conv2DTranspose layer (except the final layer) is followed by a Batch Normalization layer and a **Relu activation**; for more implementation details, check out <a href="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0XCEEN/papers/1511.06434.pdf">[2]</a>. 
- The final transpose convolution layer has three output channels since the output needs to be a color image. We use the **Tanh activation** in the final layer. 

See the illustration of the architecture from <a href="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0XCEEN/papers/1511.06434.pdf">[2]</a> below.

<center><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0XCEEN/images/generator.png" alt="generator image" width="80%"></center>


We build the Generator network by using the parameter values from <a href="https://learnopencv.com/deep-convolutional-gan-in-pytorch-and-tensorflow/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsIBMGPXX0XCEEN72-2022-01-01" >[3]<a>.



```python
def make_generator():
    
    model=Sequential()
    
    # input is latent vector of 100 dimensions
    model.add(Input(shape=(1, 1, 100), name='input_layer'))
    
    # Block 1 dimensionality of the output space  64 * 8
    model.add(Conv2DTranspose(64 * 8, kernel_size=4, strides= 4, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_1'))
    model.add(BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_1'))
    model.add(ReLU(name='relu_1'))

    # Block 2: input is 4 x 4 x (64 * 8)
    model.add(Conv2DTranspose(64 * 4, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_2'))
    model.add(BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_2'))
    model.add(ReLU(name='relu_2'))

    # Block 3: input is 8 x 8 x (64 * 4)
    model.add(Conv2DTranspose(64 * 2, kernel_size=4,strides=  2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_3'))
    model.add(BatchNormalization(momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='bn_3'))
    model.add(ReLU(name='relu_3'))

                       
    # Block 4: input is 16 x 16 x (64 * 2)
    model.add(Conv2DTranspose(64 * 1, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_4'))
    model.add(BatchNormalization(momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='bn_4'))
    model.add(ReLU(name='relu_4'))

    model.add(Conv2DTranspose(3, 4, 2,padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, 
                              activation='tanh', name='conv_transpose_5'))

    return model
```

By printing the summary of the Generator architecture, we can see that the transposed convolutions **upsample** a 100-dim input vector to a high-dimensional image of size 64 x 64 x 3.



```python
gen = make_generator()
gen.summary()
```

    Model: "sequential_5"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv_transpose_1 (Conv2DTra  (None, 4, 4, 512)        819200    
     nspose)                                                         
                                                                     
     bn_1 (BatchNormalization)   (None, 4, 4, 512)         2048      
                                                                     
     relu_1 (ReLU)               (None, 4, 4, 512)         0         
                                                                     
     conv_transpose_2 (Conv2DTra  (None, 8, 8, 256)        2097152   
     nspose)                                                         
                                                                     
     bn_2 (BatchNormalization)   (None, 8, 8, 256)         1024      
                                                                     
     relu_2 (ReLU)               (None, 8, 8, 256)         0         
                                                                     
     conv_transpose_3 (Conv2DTra  (None, 16, 16, 128)      524288    
     nspose)                                                         
                                                                     
     bn_3 (BatchNormalization)   (None, 16, 16, 128)       512       
                                                                     
     relu_3 (ReLU)               (None, 16, 16, 128)       0         
                                                                     
     conv_transpose_4 (Conv2DTra  (None, 32, 32, 64)       131072    
     nspose)                                                         
                                                                     
     bn_4 (BatchNormalization)   (None, 32, 32, 64)        256       
                                                                     
     relu_4 (ReLU)               (None, 32, 32, 64)        0         
                                                                     
     conv_transpose_5 (Conv2DTra  (None, 64, 64, 3)        3072      
     nspose)                                                         
                                                                     
    =================================================================
    Total params: 3,578,624
    Trainable params: 3,576,704
    Non-trainable params: 1,920
    _________________________________________________________________


___Building the Discriminator___


The Discriminator has five convolution layers. 

- All but the first and final Conv2D layers have Batch Normalization, since directly applying batchnorm to all layers could result in sample oscillation and model instability; 
- The first four Conv2D layers use the **Leaky-Relu activation** with a slope of 0.2. 
- Lastly, instead of a fully connected layer, the  output layer has a convolution layer with a **Sigmoid activation** function.



```python
def make_discriminator():
    
    model=Sequential()
    
    # Block 1: input is 64 x 64 x (3)
    model.add(Input(shape=(64, 64, 3), name='input_layer'))
    model.add(Conv2D(64, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='conv_1'))
    model.add(LeakyReLU(0.2, name='leaky_relu_1'))

    # Block 2: input is 32 x 32 x (64)
    model.add(Conv2D(64 * 2, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='conv_2'))
    model.add(BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_1'))
    model.add(LeakyReLU(0.2, name='leaky_relu_2'))

    # Block 3
    model.add(Conv2D(64 * 4, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='conv_3'))
    model.add(BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_2'))
    model.add(LeakyReLU(0.2, name='leaky_relu_3'))


    #Block 4
    model.add(Conv2D(64 * 8, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='conv_4'))
    model.add(BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_3'))
    model.add(LeakyReLU(0.2, name='leaky_relu_4'))


    #Block 5
    model.add(Conv2D(1, 4, 2,padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False,  
                     activation='sigmoid', name='conv_5'))

    return model 
```

By printing the summary of the Discriminator architecture, we can see that the strided convolutions **downsample** an input image of size 64 x 64 x 3.



```python
disc = make_discriminator()
disc.summary()
```

    Model: "sequential_6"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv_1 (Conv2D)             (None, 32, 32, 64)        3072      
                                                                     
     leaky_relu_1 (LeakyReLU)    (None, 32, 32, 64)        0         
                                                                     
     conv_2 (Conv2D)             (None, 16, 16, 128)       131072    
                                                                     
     bn_1 (BatchNormalization)   (None, 16, 16, 128)       512       
                                                                     
     leaky_relu_2 (LeakyReLU)    (None, 16, 16, 128)       0         
                                                                     
     conv_3 (Conv2D)             (None, 8, 8, 256)         524288    
                                                                     
     bn_2 (BatchNormalization)   (None, 8, 8, 256)         1024      
                                                                     
     leaky_relu_3 (LeakyReLU)    (None, 8, 8, 256)         0         
                                                                     
     conv_4 (Conv2D)             (None, 4, 4, 512)         2097152   
                                                                     
     bn_3 (BatchNormalization)   (None, 4, 4, 512)         2048      
                                                                     
     leaky_relu_4 (LeakyReLU)    (None, 4, 4, 512)         0         
                                                                     
     conv_5 (Conv2D)             (None, 2, 2, 1)           8192      
                                                                     
    =================================================================
    Total params: 2,767,360
    Trainable params: 2,765,568
    Non-trainable params: 1,792
    _________________________________________________________________


### Defining Loss Functions

As we discussed in the previous section, the min-max optimization problem can be formulated by minimizing the cross entropy loss for the Generator and Discriminator.  

The `cross_entropy` object is the Binary Cross Entropy loss that will be used to model the objectives of the two networks.



```python
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
```


```python
def generator_loss(Xhat):
    return cross_entropy(tf.ones_like(Xhat), Xhat)
```


```python
def discriminator_loss(X, Xhat):
    real_loss = cross_entropy(tf.ones_like(X), X)
    fake_loss = cross_entropy(tf.zeros_like(Xhat), Xhat)
    total_loss = 0.5*(real_loss + fake_loss)
    return total_loss
```

### Defining Optimizers 
 
We create two Adam optimizers for the discriminator and the generator, respectively. We pass the following arguments to the optimizers:

- learning rate of 0.0002.
- beta coefficients $\beta_1 = 0.5$ and $\beta_2 = 0.999$, which are responsible for computing the running averages of the gradients during backpropagation.



```python
learning_rate = 0.0002

generator_optimizer = tf.keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5, beta_2 = 0.999 )

discriminator_optimizer = tf.keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5, beta_2 = 0.999 )
```

    WARNING:absl:`lr` is deprecated, please use `learning_rate` instead, or use the legacy optimizer, e.g.,tf.keras.optimizers.legacy.Adam.
    WARNING:absl:`lr` is deprecated, please use `learning_rate` instead, or use the legacy optimizer, e.g.,tf.keras.optimizers.legacy.Adam.


### Create Train Step Function

As this lab is more computationally intensive than the last lab, we convert the training step into a function and then use the  @tf.function decorator, which allows the function to be "compiled" into a **callable TensorFlow graph**. This will speed up the training; for more information, read <a href="https://www.tensorflow.org/guide/function?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsIBMGPXX0XCEEN72-2022-01-01">here </a> 



```python
@tf.function

def train_step(X):
    
    #random samples it was found if you increase the  stander deviation, you get better results 
    z= tf.random.normal([BATCH_SIZE, 1, 1, latent_dim])
      # needed to compute the gradients for a list of variables.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        #generated sample 
        xhat = generator(z, training=True)
        #the output of the discriminator for real data 
        real_output = discriminator(X, training=True)
        #the output of the discriminator for fake data
        fake_output = discriminator(xhat, training=True)
        
        #loss for each 
        gen_loss= generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
      # Compute the gradients for gen_loss and generator
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    # Compute the gradients for gen_loss and discriminator
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # Ask the optimizer to apply the processed gradients
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
```

Don't be intimidated by the code above, here is a summary of what a train step accomplishes:

- First, we sample `z`, a batch of noise vectors from a normal distribution ($\mu = 1, \sigma = 1$) and feed it to the Generator.
- The Generator produces generated or "fake" images `xhat`.
- We feed real images `X` and fake images `xhat` to the Discriminator and obtain `real_output` and `fake_output` respectively as the scores.
- We calculate Generator loss `gen_loss` using the `fake_output` from Discriminator since we want the fake images to fool the Discriminator as much as possible.
- We calculate Discriminator loss `disc_loss` using both the `real_output` and `fake_output` since we want the Discriminator to distinguish the two as much as possible.
- We calculate `gradients_of_generator` and  `gradients_of_discriminator` based on the losses obtained.
- Finally, we update the Generator and Discriminator by letting their respective optimizers apply the processed gradients on the trainable model parameters.


We can transform the random noise using the generator. As the generator is not trained yet, the output appears to be noises:



```python
generator= make_generator()
BATCH_SIZE=128

latent_dim=100
noise = tf.random.normal([BATCH_SIZE, 1, 1, latent_dim])
Xhat=generator(noise,training=False)
plot_array(Xhat)
```


    
![png](output_118_0.png)
    


### Training DCGANs


As this method is computationally intensive, we will train the model for one epoch and then use the generator to produce artificial images.

__Even 1 epoch in DCGANs training takes long time.__ You can __stop the training__ here and import the pre-trained model following the instruction below.



```python
epochs=1

discriminator=make_discriminator()

generator= make_generator()


for epoch in range(epochs):
    
    #data for the true distribution of your real data samples training ste
    start = time.time()
    i=0
    for X in tqdm(normalized_ds, desc=f"epoch {epoch+1}", total=len(normalized_ds)):
        
        i+=1
        if i%1000:
            print("epoch {}, iteration {}".format(epoch+1, i))
            
        train_step(X)
    

    noise = tf.random.normal([BATCH_SIZE, 1, 1, latent_dim])
    Xhat=generator(noise,training=False)
    X=[x for x in normalized_ds]
    print("orignal images")
    plot_array(X[0])
    print("generated images")
    plot_array(Xhat)
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
```

    epoch 1:   0%|          | 0/157 [00:00<?, ?it/s]

    epoch 1, iteration 1


    epoch 1:   1%|          | 1/157 [00:08<22:29,  8.65s/it]

    epoch 1, iteration 2


    epoch 1:   1%|▏         | 2/157 [00:14<17:20,  6.71s/it]

    epoch 1, iteration 3


    epoch 1:   2%|▏         | 3/157 [00:18<15:03,  5.87s/it]

    epoch 1, iteration 4


    epoch 1:   3%|▎         | 4/157 [00:24<14:15,  5.59s/it]

    epoch 1, iteration 5


    epoch 1:   3%|▎         | 5/157 [00:29<13:41,  5.40s/it]

    epoch 1, iteration 6


    epoch 1:   4%|▍         | 6/157 [00:34<13:28,  5.36s/it]

    epoch 1, iteration 7


    epoch 1:   4%|▍         | 7/157 [00:39<13:06,  5.25s/it]

    epoch 1, iteration 8


    epoch 1:   5%|▌         | 8/157 [00:44<12:37,  5.08s/it]

    epoch 1, iteration 9


    epoch 1:   6%|▌         | 9/157 [00:49<12:33,  5.09s/it]

    epoch 1, iteration 10


    epoch 1:   6%|▋         | 10/157 [00:54<12:26,  5.08s/it]

    epoch 1, iteration 11


    epoch 1:   7%|▋         | 11/157 [00:59<12:19,  5.06s/it]

    epoch 1, iteration 12


    epoch 1:   8%|▊         | 12/157 [01:04<12:06,  5.01s/it]

    epoch 1, iteration 13


    epoch 1:   8%|▊         | 13/157 [01:10<12:43,  5.30s/it]

    epoch 1, iteration 14


    epoch 1:   9%|▉         | 14/157 [01:15<12:30,  5.25s/it]

    epoch 1, iteration 15


    epoch 1:  10%|▉         | 15/157 [01:20<12:12,  5.16s/it]

    epoch 1, iteration 16


    epoch 1:  10%|█         | 16/157 [01:25<12:13,  5.20s/it]

    epoch 1, iteration 17


    epoch 1:  11%|█         | 17/157 [01:30<12:00,  5.15s/it]

    epoch 1, iteration 18


    epoch 1:  11%|█▏        | 18/157 [01:35<11:57,  5.16s/it]

    epoch 1, iteration 19


    epoch 1:  12%|█▏        | 19/157 [01:40<11:39,  5.07s/it]

    epoch 1, iteration 20


    epoch 1:  13%|█▎        | 20/157 [01:45<11:33,  5.06s/it]

    epoch 1, iteration 21


    epoch 1:  13%|█▎        | 21/157 [01:50<11:22,  5.02s/it]

    epoch 1, iteration 22


    epoch 1:  14%|█▍        | 22/157 [01:55<11:07,  4.94s/it]

    epoch 1, iteration 23


    epoch 1:  15%|█▍        | 23/157 [02:00<11:01,  4.94s/it]

    epoch 1, iteration 24


    epoch 1:  15%|█▌        | 24/157 [02:05<11:05,  5.00s/it]

    epoch 1, iteration 25


    epoch 1:  16%|█▌        | 25/157 [02:10<11:02,  5.02s/it]

    epoch 1, iteration 26


    epoch 1:  17%|█▋        | 26/157 [02:16<11:43,  5.37s/it]

    epoch 1, iteration 27


    epoch 1:  17%|█▋        | 27/157 [02:21<11:19,  5.23s/it]

    epoch 1, iteration 28


    epoch 1:  18%|█▊        | 28/157 [02:26<11:07,  5.18s/it]

    epoch 1, iteration 29


    epoch 1:  18%|█▊        | 29/157 [02:31<10:50,  5.08s/it]

    epoch 1, iteration 30


    epoch 1:  19%|█▉        | 30/157 [02:36<10:43,  5.06s/it]

    epoch 1, iteration 31


    epoch 1:  20%|█▉        | 31/157 [02:41<10:38,  5.07s/it]

    epoch 1, iteration 32


    epoch 1:  20%|██        | 32/157 [02:46<10:31,  5.05s/it]

    epoch 1, iteration 33


    epoch 1:  21%|██        | 33/157 [02:51<10:27,  5.06s/it]

    epoch 1, iteration 34


    epoch 1:  22%|██▏       | 34/157 [02:56<10:21,  5.05s/it]

    epoch 1, iteration 35


    epoch 1:  22%|██▏       | 35/157 [03:01<10:12,  5.02s/it]

    epoch 1, iteration 36


    epoch 1:  23%|██▎       | 36/157 [03:06<10:08,  5.03s/it]

    epoch 1, iteration 37


    epoch 1:  24%|██▎       | 37/157 [03:11<10:05,  5.04s/it]

    epoch 1, iteration 38


    epoch 1:  24%|██▍       | 38/157 [03:17<10:29,  5.29s/it]

    epoch 1, iteration 39


    epoch 1:  25%|██▍       | 39/157 [03:23<10:44,  5.46s/it]

    epoch 1, iteration 40


    epoch 1:  25%|██▌       | 40/157 [03:28<10:26,  5.36s/it]

    epoch 1, iteration 41


    epoch 1:  26%|██▌       | 41/157 [03:33<10:05,  5.22s/it]

    epoch 1, iteration 42


    epoch 1:  27%|██▋       | 42/157 [03:38<09:50,  5.13s/it]

    epoch 1, iteration 43


    epoch 1:  27%|██▋       | 43/157 [03:43<09:36,  5.05s/it]

    epoch 1, iteration 44


    epoch 1:  28%|██▊       | 44/157 [03:48<09:28,  5.03s/it]

    epoch 1, iteration 45


    epoch 1:  29%|██▊       | 45/157 [03:53<09:16,  4.97s/it]

    epoch 1, iteration 46


    epoch 1:  29%|██▉       | 46/157 [03:58<09:13,  4.99s/it]

    epoch 1, iteration 47


    epoch 1:  30%|██▉       | 47/157 [04:03<09:10,  5.01s/it]

    epoch 1, iteration 48


    epoch 1:  31%|███       | 48/157 [04:08<09:05,  5.00s/it]

    epoch 1, iteration 49


    epoch 1:  31%|███       | 49/157 [04:13<08:53,  4.94s/it]

    epoch 1, iteration 50


    epoch 1:  32%|███▏      | 50/157 [04:18<08:51,  4.97s/it]

    epoch 1, iteration 51


    epoch 1:  32%|███▏      | 51/157 [04:24<09:28,  5.37s/it]

    epoch 1, iteration 52


    epoch 1:  33%|███▎      | 52/157 [04:29<09:16,  5.30s/it]

    epoch 1, iteration 53


    epoch 1:  34%|███▍      | 53/157 [04:34<08:58,  5.17s/it]

    epoch 1, iteration 54


    epoch 1:  34%|███▍      | 54/157 [04:39<08:45,  5.10s/it]

    epoch 1, iteration 55


    epoch 1:  35%|███▌      | 55/157 [04:44<08:32,  5.02s/it]

    epoch 1, iteration 56


    epoch 1:  36%|███▌      | 56/157 [04:49<08:29,  5.04s/it]

    epoch 1, iteration 57


    epoch 1:  36%|███▋      | 57/157 [04:54<08:18,  4.99s/it]

    epoch 1, iteration 58


    epoch 1:  37%|███▋      | 58/157 [04:58<08:10,  4.95s/it]

    epoch 1, iteration 59


    epoch 1:  38%|███▊      | 59/157 [05:04<08:09,  4.99s/it]

    epoch 1, iteration 60


    epoch 1:  38%|███▊      | 60/157 [05:09<08:03,  4.98s/it]

    epoch 1, iteration 61


    epoch 1:  39%|███▉      | 61/157 [05:14<08:00,  5.01s/it]

    epoch 1, iteration 62


    epoch 1:  39%|███▉      | 62/157 [05:19<07:55,  5.01s/it]

    epoch 1, iteration 63


    epoch 1:  40%|████      | 63/157 [05:23<07:47,  4.97s/it]

    epoch 1, iteration 64


    epoch 1:  41%|████      | 64/157 [05:29<08:11,  5.28s/it]

    epoch 1, iteration 65


    epoch 1:  41%|████▏     | 65/157 [05:34<07:50,  5.12s/it]

    epoch 1, iteration 66


    epoch 1:  42%|████▏     | 66/157 [05:39<07:43,  5.10s/it]

    epoch 1, iteration 67


    epoch 1:  43%|████▎     | 67/157 [05:44<07:34,  5.05s/it]

    epoch 1, iteration 68


    epoch 1:  43%|████▎     | 68/157 [05:49<07:24,  4.99s/it]

    epoch 1, iteration 69


    epoch 1:  44%|████▍     | 69/157 [05:54<07:20,  5.01s/it]

    epoch 1, iteration 70


    epoch 1:  45%|████▍     | 70/157 [05:59<07:22,  5.09s/it]

    epoch 1, iteration 71


    epoch 1:  45%|████▌     | 71/157 [06:05<07:20,  5.13s/it]

    epoch 1, iteration 72


    epoch 1:  46%|████▌     | 72/157 [06:10<07:15,  5.12s/it]

    epoch 1, iteration 73


    epoch 1:  46%|████▋     | 73/157 [06:14<07:01,  5.02s/it]

    epoch 1, iteration 74


    epoch 1:  47%|████▋     | 74/157 [06:20<06:57,  5.03s/it]

    epoch 1, iteration 75


    epoch 1:  48%|████▊     | 75/157 [06:25<06:51,  5.02s/it]

    epoch 1, iteration 76


    epoch 1:  48%|████▊     | 76/157 [06:31<07:16,  5.39s/it]

    epoch 1, iteration 77


    epoch 1:  49%|████▉     | 77/157 [06:36<07:00,  5.25s/it]

    epoch 1, iteration 78


    epoch 1:  50%|████▉     | 78/157 [06:41<06:47,  5.15s/it]

    epoch 1, iteration 79


    epoch 1:  50%|█████     | 79/157 [06:46<06:43,  5.17s/it]

    epoch 1, iteration 80


    epoch 1:  51%|█████     | 80/157 [06:51<06:33,  5.11s/it]

    epoch 1, iteration 81


    epoch 1:  52%|█████▏    | 81/157 [06:56<06:27,  5.11s/it]

    epoch 1, iteration 82


    epoch 1:  52%|█████▏    | 82/157 [07:01<06:19,  5.06s/it]

    epoch 1, iteration 83


    epoch 1:  53%|█████▎    | 83/157 [07:06<06:19,  5.13s/it]

    epoch 1, iteration 84


    epoch 1:  54%|█████▎    | 84/157 [07:11<06:15,  5.14s/it]

    epoch 1, iteration 85


    epoch 1:  54%|█████▍    | 85/157 [07:16<06:04,  5.06s/it]

    epoch 1, iteration 86


    epoch 1:  55%|█████▍    | 86/157 [07:22<06:07,  5.17s/it]

    epoch 1, iteration 87


    epoch 1:  55%|█████▌    | 87/157 [07:27<05:58,  5.12s/it]

    epoch 1, iteration 88


    epoch 1:  56%|█████▌    | 88/157 [07:32<05:48,  5.06s/it]

    epoch 1, iteration 89


    epoch 1:  57%|█████▋    | 89/157 [07:38<06:09,  5.44s/it]

    epoch 1, iteration 90


    epoch 1:  57%|█████▋    | 90/157 [07:43<05:58,  5.35s/it]

    epoch 1, iteration 91


    epoch 1:  58%|█████▊    | 91/157 [07:48<05:45,  5.24s/it]

    epoch 1, iteration 92


    epoch 1:  59%|█████▊    | 92/157 [07:53<05:33,  5.13s/it]

    epoch 1, iteration 93


    epoch 1:  59%|█████▉    | 93/157 [07:58<05:21,  5.03s/it]

    epoch 1, iteration 94


    epoch 1:  60%|█████▉    | 94/157 [08:03<05:14,  5.00s/it]

    epoch 1, iteration 95


    epoch 1:  61%|██████    | 95/157 [08:07<05:05,  4.93s/it]

    epoch 1, iteration 96


    epoch 1:  61%|██████    | 96/157 [08:12<05:01,  4.95s/it]

    epoch 1, iteration 97


    epoch 1:  62%|██████▏   | 97/157 [08:17<04:55,  4.92s/it]

    epoch 1, iteration 98


    epoch 1:  62%|██████▏   | 98/157 [08:22<04:49,  4.91s/it]

    epoch 1, iteration 99


    epoch 1:  63%|██████▎   | 99/157 [08:27<04:44,  4.90s/it]

    epoch 1, iteration 100


    epoch 1:  64%|██████▎   | 100/157 [08:32<04:40,  4.92s/it]

    epoch 1, iteration 101


    epoch 1:  64%|██████▍   | 101/157 [08:37<04:33,  4.88s/it]

    epoch 1, iteration 102


    epoch 1:  65%|██████▍   | 102/157 [08:42<04:33,  4.96s/it]

    epoch 1, iteration 103


    epoch 1:  66%|██████▌   | 103/157 [08:47<04:27,  4.95s/it]

    epoch 1, iteration 104


    epoch 1:  66%|██████▌   | 104/157 [08:52<04:23,  4.98s/it]

    epoch 1, iteration 105


    epoch 1:  67%|██████▋   | 105/157 [08:57<04:19,  5.00s/it]

    epoch 1, iteration 106


    epoch 1:  68%|██████▊   | 106/157 [09:02<04:17,  5.06s/it]

    epoch 1, iteration 107


    epoch 1:  68%|██████▊   | 107/157 [09:07<04:11,  5.03s/it]

    epoch 1, iteration 108


    epoch 1:  69%|██████▉   | 108/157 [09:12<04:08,  5.08s/it]

    epoch 1, iteration 109


    epoch 1:  69%|██████▉   | 109/157 [09:17<04:00,  5.00s/it]

    epoch 1, iteration 110


    epoch 1:  70%|███████   | 110/157 [09:22<03:54,  5.00s/it]

    epoch 1, iteration 111


    epoch 1:  71%|███████   | 111/157 [09:27<03:51,  5.04s/it]

    epoch 1, iteration 112


    epoch 1:  71%|███████▏  | 112/157 [09:32<03:46,  5.04s/it]

    epoch 1, iteration 113


    epoch 1:  72%|███████▏  | 113/157 [09:37<03:41,  5.03s/it]

    epoch 1, iteration 114


    epoch 1:  73%|███████▎  | 114/157 [09:42<03:33,  4.97s/it]

    epoch 1, iteration 115


    epoch 1:  73%|███████▎  | 115/157 [09:47<03:29,  4.98s/it]

    epoch 1, iteration 116


    epoch 1:  74%|███████▍  | 116/157 [09:52<03:21,  4.92s/it]

    epoch 1, iteration 117


    epoch 1:  75%|███████▍  | 117/157 [09:57<03:16,  4.92s/it]

    epoch 1, iteration 118


    epoch 1:  75%|███████▌  | 118/157 [10:02<03:12,  4.93s/it]

    epoch 1, iteration 119


    epoch 1:  76%|███████▌  | 119/157 [10:07<03:08,  4.96s/it]

    epoch 1, iteration 120


    epoch 1:  76%|███████▋  | 120/157 [10:12<03:03,  4.95s/it]

    epoch 1, iteration 121


    epoch 1:  77%|███████▋  | 121/157 [10:17<02:57,  4.93s/it]

    epoch 1, iteration 122


    epoch 1:  78%|███████▊  | 122/157 [10:22<02:53,  4.94s/it]

    epoch 1, iteration 123


    epoch 1:  78%|███████▊  | 123/157 [10:26<02:45,  4.87s/it]

    epoch 1, iteration 124


    epoch 1:  79%|███████▉  | 124/157 [10:31<02:40,  4.85s/it]

    epoch 1, iteration 125


    epoch 1:  80%|███████▉  | 125/157 [10:36<02:36,  4.89s/it]

    epoch 1, iteration 126


    epoch 1:  80%|████████  | 126/157 [10:41<02:34,  4.98s/it]

    epoch 1, iteration 127


    epoch 1:  81%|████████  | 127/157 [10:46<02:29,  4.98s/it]

    epoch 1, iteration 128


    epoch 1:  82%|████████▏ | 128/157 [10:51<02:23,  4.96s/it]

    epoch 1, iteration 129


    epoch 1:  82%|████████▏ | 129/157 [10:56<02:19,  4.98s/it]

    epoch 1, iteration 130


    epoch 1:  83%|████████▎ | 130/157 [11:01<02:15,  5.00s/it]

    epoch 1, iteration 131


    epoch 1:  83%|████████▎ | 131/157 [11:06<02:10,  5.03s/it]

    epoch 1, iteration 132


    epoch 1:  84%|████████▍ | 132/157 [11:11<02:03,  4.95s/it]

    epoch 1, iteration 133


    epoch 1:  85%|████████▍ | 133/157 [11:16<01:58,  4.94s/it]

    epoch 1, iteration 134


    epoch 1:  85%|████████▌ | 134/157 [11:21<01:53,  4.91s/it]

    epoch 1, iteration 135


    epoch 1:  86%|████████▌ | 135/157 [11:26<01:48,  4.91s/it]

    epoch 1, iteration 136


    epoch 1:  87%|████████▋ | 136/157 [11:31<01:43,  4.95s/it]

    epoch 1, iteration 137


    epoch 1:  87%|████████▋ | 137/157 [11:36<01:38,  4.91s/it]

    epoch 1, iteration 138


    epoch 1:  88%|████████▊ | 138/157 [11:41<01:33,  4.93s/it]

    epoch 1, iteration 139


    epoch 1:  89%|████████▊ | 139/157 [11:45<01:27,  4.85s/it]

    epoch 1, iteration 140


    epoch 1:  89%|████████▉ | 140/157 [11:50<01:22,  4.86s/it]

    epoch 1, iteration 141


    epoch 1:  90%|████████▉ | 141/157 [11:55<01:17,  4.86s/it]

    epoch 1, iteration 142


    epoch 1:  90%|█████████ | 142/157 [12:00<01:12,  4.84s/it]

    epoch 1, iteration 143


    epoch 1:  91%|█████████ | 143/157 [12:05<01:08,  4.87s/it]

    epoch 1, iteration 144


    epoch 1:  92%|█████████▏| 144/157 [12:09<01:02,  4.83s/it]

    epoch 1, iteration 145


    epoch 1:  92%|█████████▏| 145/157 [12:14<00:58,  4.87s/it]

    epoch 1, iteration 146


    epoch 1:  93%|█████████▎| 146/157 [12:19<00:53,  4.87s/it]

    epoch 1, iteration 147


    epoch 1:  94%|█████████▎| 147/157 [12:24<00:48,  4.88s/it]

    epoch 1, iteration 148


    epoch 1:  94%|█████████▍| 148/157 [12:29<00:43,  4.88s/it]

    epoch 1, iteration 149


    epoch 1:  95%|█████████▍| 149/157 [12:34<00:39,  4.91s/it]

    epoch 1, iteration 150


    epoch 1:  96%|█████████▌| 150/157 [12:39<00:34,  4.93s/it]

    epoch 1, iteration 151


    epoch 1:  96%|█████████▌| 151/157 [12:44<00:29,  4.96s/it]

    epoch 1, iteration 152


    epoch 1:  97%|█████████▋| 152/157 [12:49<00:24,  4.88s/it]

    epoch 1, iteration 153


    epoch 1:  97%|█████████▋| 153/157 [12:53<00:19,  4.85s/it]

    epoch 1, iteration 154


    epoch 1:  98%|█████████▊| 154/157 [12:58<00:14,  4.82s/it]

    epoch 1, iteration 155


    epoch 1:  99%|█████████▊| 155/157 [13:03<00:09,  4.88s/it]

    epoch 1, iteration 156


    epoch 1:  99%|█████████▉| 156/157 [13:08<00:04,  4.95s/it]

    epoch 1, iteration 157


    epoch 1: 100%|██████████| 157/157 [13:13<00:00,  5.05s/it]


    orignal images



    
![png](output_121_316.png)
    


    generated images



    
![png](output_121_318.png)
    


    Time for epoch 1 is 804.4689054489136 sec


As you can see that, with only one epoch of training and a reduced number of training images, our GAN didn't learn much information, and thus, the generator wasn't able to produce images that make sense to human eyes. There are two quick actions you can take to try to improve the results:

1. Re-train the GAN using the full dataset that has 63,632 images. 
    - To do so, simply go back to the **Loading the Dataset** section in **DCGANs**, replace the url of the dataset with "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module5/L2/cartoon_data.tgz", change the `directory` argument in `tf.keras.utils.image_dataset_from_directory` to `'cartoon_data'` in **Creating data generator** section and re-run all the cells. 
    - Note that using more training data does allows the model to learn better and perform better, but it will result in longer training time! **With 63K training images and batch size of 128, your model will train for ~497 iterations.**


2. Use a pre-trained generator model to generate images.
    - You don't need to experience the training time at all! 
    - Proceed to the next subsection to load a pre-trained model, and you will see that the generator trained with 150 epochs can produce almost realistic anime faces.


___Loading Pre-trained model (150 epochs)___


As you saw, training a GAN with only one epoch takes quite a long time. If we want to evaluate the performance of a fully trained and optimized GAN, we would need to increase the number of epochs. 
Thus, to help you **avoid extremely long training time** in this lab, we will just **download the pre-trained Generator network parameters** and then use Kera `load_model` function to obtain a **pre-trained Generator**, which we will use to generate images directly.



```python
generator_url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0XCEEN/data/generator.tar"
await skillsnetwork.prepare(generator_url, overwrite=True)
```


    Downloading generator.tar:   0%|          | 0/14632960 [00:00<?, ?it/s]



      0%|          | 0/7 [00:00<?, ?it/s]


    Saved to '.'


Load the generator:



```python
from tensorflow.keras.models import load_model


full_generator=load_model("generator")
```

    WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.


    WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.


Let's generate several images using the fully trained Generator and display them:



```python
latent_dim=100

# input consists of noise vectors
noise = tf.random.normal([200, 1, 1, latent_dim])

# feed the noise vectors to the generator
Xhat=full_generator(noise,training=False)
plot_array(Xhat)
```


    
![png](output_129_0.png)
    


## Explore Latent Variables 

Values of $\mathbf{z}$ that are relatively close together will produce similar images. For example, we can assigns elements of $\mathbf{z}$ close values such as $[1,0.8,..,0.4]$. 



```python
for c in [1,0.8,0.6,0.4]:
    Xhat=full_generator(c*tf.ones([1, 1, 1, latent_dim]),training=False) # latent_dim = 100 defined previously
    plot_array(Xhat)
```


    
![png](output_131_0.png)
    



    
![png](output_131_1.png)
    



    
![png](output_131_2.png)
    



    
![png](output_131_3.png)
    


### Exercise 1

Plot the generated images by the Generator with elements of $\mathbf{z}$ equal $[-1,-0.8,-0.6,-0.4]$.



```python
for c in [1,0.8,0.6,0.4]:
    Xhat=full_generator(-c*tf.ones([1, 1, 1, latent_dim]),training=False) # latent_dim = 100 defined previously
    plot_array(Xhat)
```


    
![png](output_133_0.png)
    



    
![png](output_133_1.png)
    



    
![png](output_133_2.png)
    



    
![png](output_133_3.png)
    


<details>
    <summary>Click here for Solution</summary>

```python
for c in [1,0.8,0.6,0.4]:
    Xhat=full_generator(-c*tf.ones([1, 1, 1, latent_dim]),training=False)
    plot_array(Xhat)
 ```   

</details>


We can see how changing the latent variable changes the generated image. Here we alter more and more subsequent values of $\mathbf{z}$ from 1 to -1; we see the images change accordingly; this is evident in the anime character's hair color:



```python
z=np.ones( (1, 1, 1, latent_dim))
for n in range(10):

    z[0, 0, 0, 0:10*n]=-1

    Xhat=full_generator(z,training=False)
    print("elements from 0 to {} is set to -1".format(10*n))
    plot_array(Xhat)
```

    elements from 0 to 0 is set to -1



    
![png](output_136_1.png)
    


    elements from 0 to 10 is set to -1



    
![png](output_136_3.png)
    


    elements from 0 to 20 is set to -1



    
![png](output_136_5.png)
    


    elements from 0 to 30 is set to -1



    
![png](output_136_7.png)
    


    elements from 0 to 40 is set to -1



    
![png](output_136_9.png)
    


    elements from 0 to 50 is set to -1



    
![png](output_136_11.png)
    


    elements from 0 to 60 is set to -1



    
![png](output_136_13.png)
    


    elements from 0 to 70 is set to -1



    
![png](output_136_15.png)
    


    elements from 0 to 80 is set to -1



    
![png](output_136_17.png)
    


    elements from 0 to 90 is set to -1



    
![png](output_136_19.png)
    


### Exercise 2

Repeat the above procedure but set the latent variable $z[0, 0, 0, 0:20*n] = -0.5*n$ each time `for n in range(5)`



```python
z=np.ones( (1, 1, 1, latent_dim))
for n in range(10):

    z[0, 0, 0, 0:10*n]=-0.5 * n

    Xhat=full_generator(z,training=False)
    print("elements from 0 to {} is set to -1".format(10*n))
    plot_array(Xhat)
```

    elements from 0 to 0 is set to -1



    
![png](output_138_1.png)
    


    elements from 0 to 10 is set to -1



    
![png](output_138_3.png)
    


    elements from 0 to 20 is set to -1



    
![png](output_138_5.png)
    


    elements from 0 to 30 is set to -1



    
![png](output_138_7.png)
    


    elements from 0 to 40 is set to -1



    
![png](output_138_9.png)
    


    elements from 0 to 50 is set to -1



    
![png](output_138_11.png)
    


    elements from 0 to 60 is set to -1



    
![png](output_138_13.png)
    


    elements from 0 to 70 is set to -1



    
![png](output_138_15.png)
    


    elements from 0 to 80 is set to -1



    
![png](output_138_17.png)
    


    elements from 0 to 90 is set to -1



    
![png](output_138_19.png)
    


<details>
    <summary>Click here for Solution</summary>

```python
z=np.ones( (1, 1, 1, latent_dim))
for n in range(5):

    z[0, 0, 0, 0:20*n]=-0.5*n

    Xhat=full_generator(z,training=False)

    plot_array(Xhat)
    

 ```   

</details>


We can also hold some of the elements of $\mathbf{z}$ constant and randomly change others. Here, we set the first 20 elements to one and randomly change the rest. We see that all through the images change, the hair color remains light.



```python
for n in range(10):
    z=np.random.normal(0, 1, (1, 1, 1, latent_dim))

    z[0,0,0,0:35]=1

    Xhat=full_generator(z,training=False)

    plot_array(Xhat)
```


    
![png](output_141_0.png)
    



    
![png](output_141_1.png)
    



    
![png](output_141_2.png)
    



    
![png](output_141_3.png)
    



    
![png](output_141_4.png)
    



    
![png](output_141_5.png)
    



    
![png](output_141_6.png)
    



    
![png](output_141_7.png)
    



    
![png](output_141_8.png)
    



    
![png](output_141_9.png)
    


### Exercise 3

Repeat the procedure above, but set the elements of $\mathbf{z}$ from index 0 to 35 to -1



```python
for n in range(10):
    z=np.random.normal(0, 1, (1, 1, 1, latent_dim))

    z[0,0,0,0:35]=-1

    Xhat=full_generator(z,training=False)

    plot_array(Xhat)
```


    
![png](output_143_0.png)
    



    
![png](output_143_1.png)
    



    
![png](output_143_2.png)
    



    
![png](output_143_3.png)
    



    
![png](output_143_4.png)
    



    
![png](output_143_5.png)
    



    
![png](output_143_6.png)
    



    
![png](output_143_7.png)
    



    
![png](output_143_8.png)
    



    
![png](output_143_9.png)
    


<details>
    <summary>Click here for Solution</summary>

```python

for n in range(10):
    z=np.random.normal(0, 1, (1, 1, 1, latent_dim))

    z[0,0,0,0:35]=-1

    Xhat=full_generator(z,training=False)

    plot_array(Xhat)
    

 ```   

</details>


__Thank you for completing this lab!__
<center><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0XCEEN/images/unknown5.jpeg" width="500px"></center>


----


## Authors


<a href="https://www.linkedin.com/in/joseph-s-50398b136/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01" target="_blank">Joseph Santarcangelo</a> has a Ph.D. in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.

[Roxanne Li](https://www.linkedin.com/in/roxanne-li/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsIBMGPXX0XCEEN72-2022-01-01) is a Data Science intern at IBM Skills Network, entering level-5 study in the Mathematics & Statistics undergraduate Coop program at McMaster University.

[Junxing(J.C.) Chen](https://www.linkedin.com/in/junxing-chen-3591a4162/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsIBMGPXX0XCEEN72-2022-01-01) is a Data Scientist at IBM with strong interests in machine learning and cutting-edge technologies.


## Change Log


| Date (YYYY-MM-DD) | Version | Changed By  | Change Description |
| ----------------- | ------- | ----------- | ------------------ |
| 2022-08-30        | 0.1     | Joseph Santarcangelo  | Created Lab       |
| 2022-09-06        | 0.1     | Roxanne Li  | Reviewed and edited Lab       |
| 2022-09-23        | 0.1     | Steve Hord  | QA pass edits                 |
| 2022-10-11        | 0.2     | Junxing(J.C.) Chen  | Reviewed and edited Lab       |


Copyright © 2022 IBM Corporation. All rights reserved.

