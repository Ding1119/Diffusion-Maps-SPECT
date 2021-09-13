# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 17:04:10 2020

@author: Ding1119
"""

# In[]



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np  
import itertools
import os





from skimage import data_dir,io,color
import torch
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import torch
from torch.autograd import Variable



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm 
import matplotlib as mpl

coll = io.ImageCollection(r'C:\Users\Ding1119\PyTorch-Spectral-clustering-master\Mask relate\a2\*.jpg')
#coll = io.ImageCollection(r'C:\Users\adm\SPECT_3_3\mask_three\*.jpg')
#coll = io.ImageCollection(r'C:\Users\adm\SPECT_3_3\all_SPECT_RGB\*.jpg')
DF=io.concatenate_images(coll)
#DF = DF[0:100]
#DF = DF[0:300]
DF = DF[0:630]/255
#DF= DF.reshape(1890,128*128*3)

#DF = preprocessing.normalize(DF)

#dff = torch.tensor(DF)




# In[]
X_train = DF

import pandas as pd
LB = pd.read_csv(r"C:\Users\Ding1119\PyTorch-Spectral-clustering-master\12_11_PPMI_test\Our_df_label_3(mid).csv")

Y_train = LB['CATEGORY_ID']
Y_train.values
Y_train.shape








# In[]

from skimage import data_dir,io,color
import torch
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import torch
from torch.autograd import Variable



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm 
import matplotlib as mpl

coll = io.ImageCollection(r'C:\Users\Ding1119\PyTorch-Spectral-clustering-master\12_11_PPMI_test\New_100\new_mid\*.jpg')
#coll = io.ImageCollection(r'C:\Users\adm\SPECT_3_3\mask_three\*.jpg')
#coll = io.ImageCollection(r'C:\Users\adm\SPECT_3_3\all_SPECT_RGB\*.jpg')
DF_=io.concatenate_images(coll)
#DF = DF[0:100]
#DF = DF[0:300]
X_test = DF_[0:100]/255
#DF= DF.reshape(1890,128*128*3)

#DF = preprocessing.normalize(DF)

#dff = torch.tensor(DF)
#X_test.shape

# In[]




import pandas as pd
LB_ = pd.read_csv(r"C:\Users\Ding1119\PyTorch-Spectral-clustering-master\12_11_PPMI_test\New_100_Label_3.csv")

Y_test = LB_['Label']
Y_test.values
#Y_test.shape



# In[]
from keras.utils.np_utils import to_categorical
y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)

train_X = X_train
test_X = X_test
print(train_X.shape)
print(y_train.shape)
print(test_X.shape)
print(y_test.shape)

# In[]




# In[]




# In[]


def split_test_LB(data):
    shuffled_indices=np.random.RandomState(seed=42).permutation(len(data))
    #test_set_size=int(len(data)*test_ratio)
    #test_indices =shuffled_indices[:test_set_size]
    train_indices_1=shuffled_indices[0:130]
    train_indices_2=shuffled_indices[130:630]
    #train_indices_3=shuffled_indices[8:12]
    #train_indices_4=shuffled_indices[12:16]
    #train_indices_5=shuffled_indices[16:20]
    
    
    return data[train_indices_1],data[train_indices_2]



# In[]

def split_test(data):
    shuffled_indices=np.random.RandomState(seed=42).permutation(len(data))
    #test_set_size=int(len(data)*test_ratio)
    #test_indices =shuffled_indices[:test_set_size]
    train_indices_1=shuffled_indices[0:130]
    train_indices_2=shuffled_indices[130:630]
    #train_indices_3=shuffled_indices[8:12]
    #train_indices_4=shuffled_indices[12:16]
    #train_indices_5=shuffled_indices[16:20]
    
    
    return data[train_indices_1],data[train_indices_2]




# In[]


all_LB = split_test_LB(y_train)
all_split_df = split_test(train_X)
DF_y_test = all_LB[0]
DF_X_test = all_split_df[0]
DF_X_train = all_split_df[1]
DF_y_train = all_LB[1]



# In[]



#X = np.array([[1, 2], [3, 4], [5, 6], [7, 8],[9,10],[11,12]])
#y = np.array([1, 2, 3, 4,5,6])
#train_X = np.zeros((6,2))
#X = list(X)
from sklearn.model_selection import KFold
X = DF_X_train
y = DF_y_train
train_X = []
Y_train =[]
test_X = []
Y_test = []
from sklearn.model_selection import RepeatedKFold
kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=8)

#kf =  KFold(25)
#kf = KFold(n_splits=25)
for train_index, test_index in kf.split(X,y):
    train_X.append(X[train_index])
    Y_train.append(y[train_index])
    test_X.append(X[test_index])
    Y_test.append(y[test_index])
    print('train_index', train_index, 'test_index', test_index)







# In[]


for train_index in range(25):
    print('第',train_index,'個 Fold')
    print('train_X shape:',train_X[train_index].shape)
    print('train_Y shape:',Y_train[train_index].shape)
    print('test_X shape:',test_X[train_index].shape)
    print('test_Y shape:',Y_test[train_index].shape)





# In[]


shear_range = 0.1 #@param {type:"slider", min:0.0, max:1.0, step:0.05}
zoom_range = 0.1 #@param {type:"slider", min:0.0, max:1.0, step:0.05}
width_shift_range = 0.1 #@param {type:"slider", min:0.0, max:1.0, step:0.05}
height_shift_range = 0.1 #@param {type:"slider", min:0.0, max:1.0, step:0.05}
rotation_range = 10 #@param {type:"slider", min:0, max:90, step:5}
horizontal_flip = True #@param {type:"boolean"}
vertical_flip = False #@param {type:"boolean"}


# In[]
from sklearn.metrics import precision_score
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications import ResNet50
#from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.metrics import classification_report

n_folds = 25
from sklearn.model_selection import train_test_split

EPOCHS=10
BATCH_SIZE=2

#save the model history in a list after fitting so that we can plot later
model_history = [] 

TABLE = []
#model_history = []

accuracy1_ = []
sensitivity1_ = []
specificity1_ = []
precision_ = []


for i in range(n_folds):
    print("Training on Fold: ",i+1)
    
    
    X_train_1 = train_X[i]
    
    
    #X_test_1 = test_X[k].reshape((126,3*49152))
    
    X_test_1 = DF_X_test
    
    Y_train_1 =Y_train[i]
    #Y_test_1 = Y_test[k]
    
    Y_test_1 = DF_y_test
    
    
    
    
    
    
    t_x = X_train_1
    val_x =  X_test_1
    
    
    
    t_y = Y_train_1
    val_y = Y_test_1
    
    
    ################ 增強 ################
    
    aug = ImageDataGenerator(   rotation_range=30,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.2,
        height_shift_range=0.2,)

# train the network
    
    
    
    
    ###  model VGG ########
    
#     covn_base = tf.keras.applications.vgg16.VGG16(weights='imagenet',include_top=False)
#     covn_base = tf.keras.applications.resnet50.ResNet50(include_top=True, weights=None,
#                   input_shape=(128,128,3))

    
# # #     covn_base = tf.keras.applications.InceptionV3(weights='imagenet',include_top=False,input_shape=(128,128,3))
#     covn_base.trainable = True
# # #冻结前面的层，训练最后四层
#     for layers in covn_base.layers[:-4]:
        
#         layers.trainable = False
# #构建模型
#     model = tf.keras.Sequential()
#     model.add(covn_base)
#     model.add(tf.keras.layers.GlobalAveragePooling2D())
    
#     model.add(tf.keras.layers.Dense(256, activation='relu'))
#     model.add(tf.keras.layers.Dropout(0.2))
    
#     model.add(tf.keras.layers.Dense(256, activation='relu'))
#     model.add(tf.keras.layers.Dropout(0.2))
#     model.add(tf.keras.layers.Dense(128, activation='relu'))
#     model.add(tf.keras.layers.Dropout(0.2))
#     model.add(tf.keras.layers.Dense(3, activation='softmax'))

    
    # Alexnet

    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1028, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1028, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')])
    

     
    
    #model.summary()

    # model = tf.keras.models.Sequential([
    # tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', 
    #                               input_shape=(128,128,3)),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.MaxPool2D(pool_size=2) ,
    # tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.MaxPool2D(pool_size=2),
    # tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.MaxPool2D(pool_size=2),
    # tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(3, activation="softmax")])
    
    

    
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), #使用adam优化器，学习率为0.0001
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), #交叉熵损失函数
              metrics=["accuracy"]) #评价函数
    

    
    
    
    
    #t_x, val_x, t_y, val_y = train_test_split(train_X, y_train, test_size=0.1, 
                                              # random_state = np.random.randint(1,1000, 1)[0])
    #model_history.append(fit_and_evaluate(t_x, val_x, t_y, val_y, epochs, batch_size))
    
    
    
    
    
    
    ###########################################
    
    history = model.fit(t_x, t_y, epochs=EPOCHS, batch_size=BATCH_SIZE, 
             verbose=1, validation_split=0.1)
    
    #history = model.fit_generator(aug.flow(t_x, t_y, batch_size=BATCH_SIZE),
                       # validation_data=(val_x, val_y), steps_per_epoch=len(t_x) // BATCH_SIZE,
                        #epochs=EPOCHS)
    
    
   # model_history.append(model.evaluate(val_x, val_y))
    
    y_pred = model.predict_classes(val_x )
    
    #print(y_pred)
              
    
    val_y = np.argmax(val_y ,axis=1)
    
    print(classification_report(y_true=val_y, y_pred=y_pred))
    
    #TABLE.append(y_pred)
    
    
    cm1 = confusion_matrix(y_true=val_y, y_pred=y_pred)
    print('Confusion Matrix : \n', cm1)

    total1=sum(sum(cm1))
#####from confusion matrix calculate accuracy
    accuracy1_.append((cm1[0,0]+cm1[1,1])/total1)
    accuracy1 = (cm1[0,0]+cm1[1,1])/total1
    print ('Accuracy : ', accuracy1)

    sensitivity1_.append(cm1[0,0]/(cm1[0,0]+cm1[0,1])) 
    sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    print('Sensitivity : ', sensitivity1 )
    
    specificity1_.append(cm1[1,1]/(cm1[1,0]+cm1[1,1]))

    specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1]) 
    print('Specificity : ', specificity1)
    
    
    precision = precision_score(y_true=val_y, y_pred=y_pred, average='weighted')
    precision_.append(precision)
    
    print('=============================================')
    #U_test = metrics.accuracy_score(y_true=val_y, y_pred=y_pred)
    
    
    #print(U_test)
    #VGG_accuracy[i] = U_test
    #
    
    ##########################################
   # model.evaluate(test_x, Y_test_1)
    
    #print("======="*12, end="\n\n\n")
    
    
    
    #Confusion matrix, Accuracy, sensitivity and specificity


    #cm1 = confusion_matrix(y_true=val_y, y_pred=y_pred)
    #print('Confusion Matrix : \n', cm1)

    #total1=sum(sum(cm1))
#####from confusion matrix calculate accuracy
    #accuracy1=(cm1[0,0]+cm1[1,1])/total1
    #print ('Accuracy : ', accuracy1)

    #sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    #print('Sensitivity : ', sensitivity1 )

    #specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    #print('Specificity : ', specificity1)


#np.save('accuracy_25_SPECT_DCNN.npy',accuracy1_) 
#np.save('sensitivity_25_SPECT_DCNN.npy',sensitivity1_) 
#np.save('specificity_25_SPECT_DCNN.npy',specificity1_)
#np.save('precision_25_SPECT_DCNN.npy',precision_)


# In[]












# In[]










# In[]








# In[]









# In[]










# In[]









# In[]










# In[]





# In[]



