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
from keras.models import load_model

from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.utils import plot_model

wd = os.getcwd()
model_path = os.path.join(wd,'model')
model_filepath = os.path.join(model_path,'x_model_1.h5')
csv_filepath = os.path.join(model_path,'x_model_1.csv')
ae_x_history = pd.read_csv(csv_filepath)

plt.figure(figsize=(10,6))
plt.plot(ae_x_history['loss'], label='Training Loss', alpha=0.4)
plt.plot(ae_x_history['val_loss'], label='Validation Loss', alpha=0.4)
plt.legend()
plt.title('Training History of X direction')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

x_autoencoder = load_model(os.path.join(model_path, 'x_model_1.h5'))

x_te1 = pd.read_csv(wd + '\\Data\\x_te1.csv', delimiter=',', header=None)
x_te2 = pd.read_csv(wd + '\\Data\\x_te2.csv', delimiter=',', header=None)
x_te3 = pd.read_csv(wd + '\\Data\\x_te3.csv', delimiter=',', header=None)

#testing data numpy array and scale
x_te1 = np.array(x_te1)
x_te1 = x_te1.astype('float32')/x_te1.max()
x_te2 = np.array(x_te2)
x_te2 = x_te2.astype('float32')/x_te2.max()
x_te3 = np.array(x_te3)
x_te3 = x_te3.astype('float32')/x_te3.max()


x_te1_predicted = x_autoencoder.predict(x_te1)
x_te2_predicted = x_autoencoder.predict(x_te2)
x_te3_predicted = x_autoencoder.predict(x_te3)

x_te1_mse = [mse(i,j) for i,j in zip(x_te1,x_te1_predicted)]
x_te2_mse = [mse(i,j) for i,j in zip(x_te2,x_te2_predicted)]
x_te3_mse = [mse(i,j) for i,j in zip(x_te3,x_te3_predicted)]


labels = np.concatenate([np.array([1] * len(x_te1_mse)),np.array([0]*len(x_te2_mse))])
mse_scores = np.concatenate([np.array(x_te1_mse),np.array(x_te2_mse)])
df_te1_te2 = pd.DataFrame({'labels': labels,'mse_scores':mse_scores})

normal_df = df_te1_te2.loc[df_te1_te2['labels']==0]
anomalous_df = df_te1_te2.loc[df_te1_te2['labels']==1]

plt.figure(figsize=(10,6))
plt.scatter(normal_df.index, normal_df['mse_scores'], label = 'normal')
plt.scatter(anomalous_df.index, anomalous_df['mse_scores'], label = 'anomalous')
plt.title('TE-1 TE-2 X Direction')
#plt.hlines(0.0009, plt.xlim()[0], plt.xlim()[1], colors= 'r', label='Threshold')
plt.xlabel('Observations')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.show()

y_pred = [0 if i<0.0009 else 1 for i in df_te1_te2['mse_scores']]
y_true = [i for i in df_te1_te2['labels']]

cm = confusion_matrix(y_true,y_pred)
cm_df = pd.DataFrame(cm,index=['normal','anomalous'],columns=['normal','anomalous'])
sns.heatmap(cm_df, annot=True, fmt='4d' )
plt.title('TE-1 TE-2 X Direction Confusion Matrix')
plt.show()

cr = classification_report(y_true,y_pred)
mcc = matthews_corrcoef(y_true,y_pred)
print(cr)
print('Matthews correlation co-efficient for TE1 TE2 X Direction is {}'.format(mcc))


### te3

df_labeled = pd.read_csv(wd + '\\Data\\label_binary.csv')
df_labeled['x_te3_mse'] = x_te3_mse
normal_df_te3 = df_labeled.loc[df_labeled['y_true']==0]
anomalous_df_te3 = df_labeled.loc[df_labeled['y_true']==1]

plt.figure(figsize=(10,6))
plt.scatter(normal_df_te3.index, normal_df_te3['x_te3_mse'], label = 'normal')
plt.scatter(anomalous_df_te3.index, anomalous_df_te3['x_te3_mse'], label = 'anomalous')
plt.title('TE-3 X Direction')
#plt.hlines(0.00097, plt.xlim()[0], plt.xlim()[1], colors= 'r', label='Threshold')
plt.hlines(0.0025, plt.xlim()[0], plt.xlim()[1], colors= 'r', label='Threshold')
plt.xlabel('Observations')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.show()

y_true_te3 = [i for i in df_labeled['y_true']]
y_pred_te3 = [1 if i<0.0025 else 0 for i in df_labeled['x_te3_mse']]

cm_te3 = confusion_matrix(y_true_te3,y_pred_te3)
cm_df_te3 = pd.DataFrame(cm_te3,index=['normal','anomalous'],columns=['normal','anomalous'])
sns.heatmap(cm_df_te3, annot=True, fmt='4d' )
plt.title('TE-3 X Direction Confusion Matrix')
plt.show()

cr_te3 = classification_report(y_true_te3,y_pred_te3)
mcc_te3 = matthews_corrcoef(y_true_te3,y_pred_te3)
print(cr_te3)
print('Matthews correlation co-efficient for TE3 X Direction is {}'.format(mcc_te3))

