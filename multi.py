
# coding: utf-8

# In[1]:

import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Merge
from keras.layers import Convolution1D ,AveragePooling1D
from keras.utils.np_utils import to_categorical


# In[2]:

train=open("Downloads/Gun_Point_TRAIN")
test=open("Downloads/Gun_Point_TEST")
train = train.read().split()
test = test.read().split()


# In[3]:

train = np.array_split(np.array(train),50)
test = np.array_split(np.array(test),150)
print(np.shape(train))
print(np.shape(test))


# In[4]:

#get the label of train and test
def Get_label(data):
    label=[]
    for i in range(0,len(data)):
        label.append(float(data[i][0]))
    return label
y_train = Get_label(train)
y_test= Get_label(test)


# In[5]:

#get the value of train and test
def Get_value(data):
    value=[]
    for i in range(0,len(data)):
        value.append(data[i][1:151])
    return value
X_train=Get_value(train)
X_test=Get_value(test)


# In[6]:

#set the series 
length=150
X_train=np.asarray(X_train)
X_test=np.asarray(X_test)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = np.asarray(y_train)
y_train= y_train.astype(int)
y_test = np.asarray(y_test)
y_test= y_test.astype(int)
y_test=y_test-1
y_train=y_train-1
X_train=X_train.reshape(50,150,1)
X_test=X_test.reshape(150,150,1)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


# In[7]:

def process_output(labels):
    return to_categorical(labels, nb_classes=2)
y_train = process_output(y_train)
y_test = process_output(y_test)


# In[8]:

print('Building model...')
#first model
model = Sequential()
#1st convolution
model.add(Convolution1D(50,8,input_shape=(150, 1)))
model.add(Activation('relu'))
model.add(AveragePooling1D(pool_length=2))
#2st convolution
model.add(Convolution1D(100,8))
model.add(Activation('relu'))
model.add(AveragePooling1D(pool_length=2))
model.add(Flatten())
#fulley connected layer
model.add(Dense(500,activation="relu"))
model.add(Dense(2))
#second model
model2 = Sequential()
#1st convolution
model2.add(Convolution1D(50,8,input_shape=(150, 1)))
model2.add(Activation('relu'))
model2.add(AveragePooling1D(pool_length=2))
#2st convolution
model2.add(Convolution1D(100,8))
model2.add(Activation('relu'))
model2.add(AveragePooling1D(pool_length=2))
model2.add(Flatten())
#fulley connected layer
model2.add(Dense(500,activation="relu"))
model2.add(Dense(2))
#merge model
merged = Merge([model, model2], mode='concat')

final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(2, activation='softmax'))

final_model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
print(final_model.summary())


# In[9]:

#input two data.  GET ERROR
#X_train to CNN1 
#X_test to CNN2
batch_size = 50
nb_epoch = 2
final_model.fit([X_train,X_train], y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          )


# In[ ]:



