import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import keras as k
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import matthews_corrcoef
from keras.layers import Input, Dense
from keras.models import Model

from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau


from keras.utils import plot_model

#x_tr = Training file
#x_te1 = All Anomalous Data
#x_te2 = Good Data
#x_te3 = Mixed Data (233, out of 2849 are anomalous)

wd = os.getcwd()
x_tr = pd.read_csv(wd + '\\Data\\x_tr.csv', delimiter=',', header=None)
x_te1 = pd.read_csv(wd + '\\Data\\x_te1.csv', delimiter=',', header=None)
x_te2 = pd.read_csv(wd + '\\Data\\x_te2.csv', delimiter=',', header=None)
x_te3 = pd.read_csv(wd + '\\Data\\x_te3.csv', delimiter=',', header=None)


x_train = np.array(x_tr)
x_train = x_train.astype('float32')/x_train.max()

x_te1 = np.array(x_te1)
x_te1 = x_te1.astype('float32')/x_te1.max()
x_te2 = np.array(x_te2)
x_te2 = x_te2.astype('float32')/x_te2.max()
x_te3 = np.array(x_te3)
x_te3 = x_te3.astype('float32')/x_te3.max()


ipfeatures = Input(shape=(256,))
encoded = Dense(128, activation='relu')(ipfeatures)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='sigmoid')(decoded)

autoencoder = Model(ipfeatures, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#graphical representation
plot_model(autoencoder, show_shapes=True, show_layer_names= True)

autoencoder.summary()

os.mkdir(os.path.join(wd,'model'))
model_path = os.path.join(wd,'model')
model_filepath = os.path.join(model_path,'model_1.h5')
csv_filepath = os.path.join(model_path,'model_1.csv') #training history of model 1 is saved here

stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min', restore_best_weights=True)
checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
csv_logs = CSVLogger(csv_filepath, separator=',', append=True)
plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='min', min_lr=0.00001)

history = autoencoder.fit(x_train, x_train,
                epochs=1000,
                batch_size=256,
                shuffle=True,
                validation_split= 0.2,
                callbacks = [checkpoint, csv_logs, stop, plateau],
                verbose=1)


x_te1_predicted = autoencoder.predict(x_te1)
x_te2_predicted = autoencoder.predict(x_te2)
x_te3_predicted = autoencoder.predict(x_te3)

x_te1_mse = [mse(i,j) for i,j in zip(x_te1,x_te1_predicted)]
x_te2_mse = [mse(i,j) for i,j in zip(x_te2,x_te2_predicted)]
x_te3_mse = [mse(i,j) for i,j in zip(x_te3,x_te3_predicted)]

plt.figure(figsize=(10,6))
plt.hist(x_te1_mse, alpha=0.5, label='anomalous')
plt.hist(x_te2_mse, alpha=0.5, label='normal')
plt.xlabel('Reconstruction Error')
plt.ylabel('Observations')
plt.legend()
plt.show()

x_te1_mse_df = pd.DataFrame({'mse':x_te1_mse})
x_te2_mse_df = pd.DataFrame({'mse':x_te2_mse})
plt.figure(figsize=(10,6))
plt.scatter(x_te1_mse_df.index, x_te1_mse_df['mse'], label = 'anomalous')
plt.scatter(x_te2_mse_df.index, x_te2_mse_df['mse'], label = 'normal')
plt.hlines(0.0009, plt.xlim()[0], plt.xlim()[1], colors= 'r', label='Threshold')
plt.xlabel('Observations')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.show()

y_true = np.concatenate([np.array([1]*len(x_te1_mse)), np.array([0]*len(x_te2_mse))])
y_pred = np.concatenate([np.array(x_te1_mse),np.array(x_te2_mse)])
y_pred = np.array([0 if i<0.001 else 1 for i in y_pred])

cm = confusion_matrix(y_true,y_pred)
cm_df = pd.DataFrame(cm,index=['normal','anomalous'],columns=['normal','anomalous'])
sns.heatmap(cm_df, annot=True, fmt='4d' )
plt.show()

cr = classification_report(y_true,y_pred)
mcc = matthews_corrcoef(y_true,y_pred)
print(cr)
print('Matthews correlation co-efficient is {}'.format(mcc))

x_te3_time = pd.read_csv(wd + '\\Data\\x_te3_time_label.csv')
x_te3_time['mse'] = x_te3_mse


x_te3_normal = x_te3_time.loc[x_te3_time['y_true']==0]
x_te3_anomalous = x_te3_time.loc[x_te3_time['y_true']==1]

plt.figure(figsize=(10,6))
plt.scatter(x_te3_normal.index, x_te3_normal['mse'], label = 'normal')
plt.scatter(x_te3_anomalous.index, x_te3_anomalous['mse'], label = 'anomalous')
plt.legend()
plt.show()

##########################################################################




bce = tf.keras.losses.BinaryCrossentropy()

bce_normal =  [bce(i,j) for i,j in zip(x_test_nonanomalous,x_test_nonanomalous_predicted)]
bce_anomalous =  [bce(i,j) for i,j in zip(x_test_anomalous,x_test_anomalous_predicted)]

bce_normal = tf.keras.losses.BinaryCrossentropy(x_test_nonanomalous,x_test_nonanomalous_predicted)
bce_anomalous = tf.keras.losses.BinaryCrossentropy(x_test_anomalous,x_test_anomalous_predicted)
plt.figure(figsize=(10,6))
plt.hist(bce_normal, alpha=0.5, label='normal')
plt.hist(bce_anomalous, alpha=0.5, label='anomalous')
plt.legend()
plt.show()






y_true_normal = np.array([0] * 4132)
y_true_anomalous = np.array([1] * 4517)
y_true = np.concatenate([y_true_normal,y_true_anomalous])

x_pred = np.concatenate([np.array(mse_normal),np.array(mse_anomalous)])
y_pred = np.array([0 if i<0.001 else 1 for i in x_pred])

cm = confusion_matrix(y_true,y_pred)
cm_df = pd.DataFrame(cm,index=['normal','anomalous'],columns=['normal','anomalous'])

sns.heatmap(cm_df, annot=True, fmt='4d' )
plt.show()

cr = classification_report(y_true,y_pred)

mcc = matthews_corrcoef(y_true,y_pred)




