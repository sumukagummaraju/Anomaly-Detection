import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fft
from functions import splitTo4, locMax, list2tuple
from functions import h2i
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# read the data in
wd = os.getcwd()
df = pd.read_csv(wd + '\\Data\\BHKW\\timeseries_f0f8f221e453lcsstream.csv', delimiter=',', header=None)
#df = pd.read_csv(wd + '\\Data\\timeseries_f0f8f221e453lcsstream_test.csv', delimiter=',', header=None)

df = df.drop(df.columns[[0, 5, 6]], axis=1)
df.columns = ['timestamp', 'UUID','streamlength','RawStreamData']
df['timestamp']=pd.to_datetime(df['timestamp'])
df= df.set_index('timestamp')
df['RawStreamData'] = df['RawStreamData'].str[2:]

#df1=df.loc[df['UUID'] == 'B000'] audio
df2 = df.loc[df['UUID'] == 'B080'] #acceleration x
df3 = df.loc[df['UUID'] == 'B081'] #acceleration y
df4 = df.loc[df['UUID'] == 'B082'] #acceleration z



"""
#check the number of unique streamlengths (USL) in B000
uslb000 = df1["streamlength"].unique()
print(uslb000)
uslb080 = df2["streamlength"].unique()
uslb081 = df3["streamlength"].unique()
uslb082 = df4["streamlength"].unique()
"""

"""
#for B000
b000_orig_rsd = df1['RawStreamData'].values.tolist()

#hexlist is a list containing every hexstring split into 2 Bytes
df1_hexlist=[]
for elem in b000_orig_rsd:
    df1_hexlist.append(splitTo4(elem))

#for every record in column rawstreamdata
#create
# 1. another column with a list of hex values split into 4 characters
# 2. another list with the integer values

#intlist is a list containing the hexstring as an integer list
df1_intlist = []
#df1_mean_intlist= []
for ele1 in df1_hexlist:
    templist = []
    for ele2 in ele1:
        templist.append(h2i(ele2))
    df1_intlist.append(templist)

df1['IntStreamData'] = df1_intlist
df1_sl = df1['streamlength'].values.tolist()

"""

"""Calculating the mean of a given stream of list
#counter = 0
#for ele in df1_intlist:
#    if len(ele)==0:
#        df1_mean_intlist.append(0)
#        #print("{} list empty".format(counter))
#    else:
#        df1_mean_intlist.append(mean(ele))
#    #counter+=1

#df1['MeanIntList'] = df1_mean_intlist
"""

"""#working model - configure a loop to go over the entire list of data points
from scipy.fft import fft, ifft
import numpy as np
import matplotlib.pyplot as plt

fourier_list=[]
xf_list = []
count_df1_sl = 0

for item in df1_sl:
    N = item
    T = 1/16000
    x = np.linspace(0,N*T,N)
    y = np.array(df1_intlist[count_df1_sl])
    yf = fft(y)
    fourier_list.append(yf)
    xf = np.linspace(0, 1 / (2 * T), N // 2)
    xf_list.append(xf)
    count_df1_sl = count_df1_sl + 1
    if count_df1_sl>3:
        break
    plt.stem(xf, 2.0 / N * np.abs(yf[0:N // 2]), linefmt='-')
    plt.xlabel('Frequency (Hz)'), plt.ylabel('Amplitude')
    # plt.xticks(np.arange(min(xf), max(xf), 20))
    # plt.yticks(0,1)
    plt.xlim(0, 4000)
    plt.ylim(0, 1)
    plt.show()

#working model for one data point
from scipy.fft import fft, ifft
import numpy as np
import matplotlib.pyplot as plt

N = 528    #Number of datapoints in one stream
T = 1.0/16000.0     #frequency in Hz
x = np.linspace(0,N*T,N)    #Time discretization
y = np.array(df1_intlist[0])    #Array of 1 stream values, consisting of 512 or 720 or values
yf = fft(y)     #Fourier transform of that particular 1 stream of values
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)    #Consider frequency only upto Nyquist
'''xmin = 0
xmax = 8000
ymin = 0
ymax = 2000
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2])) #plot only the magnitude (real value) and not the angle(phase)
plt.axis([xmin,xmax,ymin,ymax])
plt.grid()
plt.show()
'''

plt.stem(xf, 2.0/N * np.abs(yf[0:N//2]),linefmt='-')
plt.xlabel('Frequency (Hz)'), plt.ylabel('Amplitude')
#plt.xticks(np.arange(min(xf), max(xf), 20))
#plt.yticks(0,1)
plt.xlim(0, 4000)
plt.ylim(0,5)
plt.show()

real_yf = complex2real(yf) #convert 528 complex numbers to real
normalized_real_yf = normalize(real_yf) #normalize these real values between 0 and 1
lm_yf = localmaxima(yf,len(yf))

"""


#for B080 (as list)
b080_orig_rsd = df2['RawStreamData'].values.tolist()

#hexlist is a list containing every hexstring split into 2 Bytes
df2_hexlist=[]
for elem in b080_orig_rsd:
    df2_hexlist.append(splitTo4(elem))

#for every record in column rawstreamdata
#create
# 1. another column with a list of hex values split into 4 characters
# 2. another list with the integer values

#intlist is a list containing the hexstring as an integer list
df2_intlist = []
#df2_mean_intlist= []
for ele1 in df2_hexlist:
    templist = []
    for ele2 in ele1:
        templist.append(h2i(ele2))
    df2_intlist.append(templist)

df2_hexlist.clear()

#df2['IntStreamData'] = df2_intlist
#df2_sl = df2['streamlength'].values.tolist()

"""df2_fftlist = []
for elex in df2_intlist:
    tlist = []
    for eley in elex:
        x = np.array(df2_intlist[eley])
        xf = fft(x)
        xf = xf.tolist()
        tlist.append(xf)
    df2_fftlist.append(tlist) delete after working"""

#compute fourier transform of all the integer values
df2_fftlist = []
df2_intlist = np.array(df2_intlist)
for el in df2_intlist:
    temp = fft(el)
    temp = temp.tolist()
    df2_fftlist.append(temp)
    #tlist = []
    #tlist = np.empty([512])
    #for e in el:
        #x = np.array(df2_intlist[eley])
        #xf = fft(e)
        #xf = xf.tolist()
        #tlist.append(xf)
    #df2_fftlist.append(tlist)

#consider only first half of positive frequencies
df2_fftlist_half = []
for elem in df2_fftlist:
    len = elem.__len__()
    mid = len//2
    half_list = elem[:mid]
    df2_fftlist_half.append(half_list)

#calculate power  spectral density for  first half of positive frequencies
df2_fftlist_psd = []
for elem in df2_fftlist_half:
    tlist = []
    for ele in elem:
        ele = np.real(ele * np.conj(ele))
        tlist.append(ele)
    df2_fftlist_psd.append(tlist)

#save memory df2_fftlist.clear()
#save memory df2_fftlist_half.clear()

train_10k_x = pd.DataFrame(df2_fftlist_psd[0:10000], columns=np.arange(0,256,1))
train_10k_x.to_csv(wd + '\\Data\\BHKW\\train_10k_x.csv', header=None, index=None)
test_2k_x = pd.DataFrame(df2_fftlist_psd[10000:12000], columns=np.arange(0,256,1))
test_2k_x.to_csv(wd + '\\Data\\BHKW\\test_2k_x.csv', header=None, index=None)
#train_2k = pd.DataFrame(df2_fftlist_psd[0:2000], columns=np.arange(0,256,1))
#train_all = pd.DataFrame(df2_lmlist, columns=np.arange(0,256,1))
df12k_x = pd.DataFrame(df2_fftlist_psd[0:12000], columns=np.arange(0,256,1))
df12k_x.to_csv(wd + '\\Data\\BHKW\\12k_x.csv', header=None, index=None)


"""#normalization column-wise
x = train_2k[np.arange(0,256,1).tolist()].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
train_top2000_norm = pd.DataFrame(x_scaled)

train_top2000_norm.to_csv(wd + '\\Data\\BHKW\\C_Norm_Trdata.csv', header=None, index=None)"""

"""#local maxima and minima
#create another dataframe with local maxima and minima values
df2_lmlist = []
for elem in df2_fftlist_psd:
    telist = locMax(elem)
    df2_lmlist.append(telist)

#headings = np.arange(0,256,1).tolist()
#df2_top2000 = pd.DataFrame(df2_lmlist[0:2000], columns=np.arange(0,256,1))
#df2_new = pd.concat([df2,df2_attach])
#df2_train = df2_attach.head(200)
#df2_top2000.to_csv(wd + '\\Data\\BHKW\\reducedtrainingdata.csv')

df2_train = df2_lmlist[0:250]
#df2_numpy = np.asarray(df2_t)"""


"""
#visualize the power spectral density at a given instant
N = 512    #Number of datapoints in one stream
T = 1.0/6664     #frequency in Hz
xf = np.linspace(0.0, 1.0/(2.0*T), N//2) #Divide  into
yf = normalize(df2_fftlist_psd[0])

#plot the above numbers
plt.stem(xf,yf,linefmt='-')
plt.xlabel('Frequency(Hz)'), plt.ylabel('Power')
plt.xlim(0, 3332)
plt.ylim(0,0.05)
plt.show()
#plt.xticks(np.arange(min(xf), max(xf), 20))
#plt.yticks(0,1)"""

"""df2_train_np = []
for elem in df2_t:
    elem = np.asarray(df2_t)
    df2_t_np.append(elem)

df2_tuple_np = list2tuple(df2_t_np)"""

#for B081 (as list)
b081_orig_rsd = df3['RawStreamData'].values.tolist()

#hexlist is a list containing every hexstring split into 2 Bytes
df3_hexlist=[]
for elem in b081_orig_rsd:
    df3_hexlist.append(splitTo4(elem))

#for every record in column rawstreamdata
#create
# 1. another column with a list of hex values split into 4 characters
# 2. another list with the integer values

#intlist is a list containing the hexstring as an integer list
df3_intlist = []
#df2_mean_intlist= []
for ele1 in df3_hexlist:
    templist = []
    for ele2 in ele1:
        templist.append(h2i(ele2))
    df3_intlist.append(templist)

df3_hexlist.clear()

#df2['IntStreamData'] = df2_intlist
#df2_sl = df2['streamlength'].values.tolist()

"""df2_fftlist = []
for elex in df2_intlist:
    tlist = []
    for eley in elex:
        x = np.array(df2_intlist[eley])
        xf = fft(x)
        xf = xf.tolist()
        tlist.append(xf)
    df2_fftlist.append(tlist) delete after working"""


df3_fftlist = []
df3_intlist = np.array(df3_intlist)
for el in df3_intlist:
    temp = fft(el)
    temp = temp.tolist()
    df3_fftlist.append(temp)
    #tlist = []
    #tlist = np.empty([512])
    #for e in el:
        #x = np.array(df2_intlist[eley])
        #xf = fft(e)
        #xf = xf.tolist()
        #tlist.append(xf)
    #df2_fftlist.append(tlist)


df3_fftlist_half = []
for elem in df3_fftlist:
    len = elem.__len__()
    mid = len//2
    half_list = elem[:mid]
    df3_fftlist_half.append(half_list)

df3_fftlist_psd = []
for elem in df3_fftlist_half:
    tlist = []
    for ele in elem:
        ele = np.real(ele * np.conj(ele))
        tlist.append(ele)
    df3_fftlist_psd.append(tlist)

#save memory df3_fftlist.clear()
#save memory df3_fftlist_half.clear()


train_10k_y = pd.DataFrame(df3_fftlist_psd[0:10000], columns=np.arange(0,256,1))
train_10k_y.to_csv(wd + '\\Data\\BHKW\\train_10k_y.csv', header=None, index=None)
test_2k_y = pd.DataFrame(df3_fftlist_psd[10000:12000], columns=np.arange(0,256,1))
test_2k_y.to_csv(wd + '\\Data\\BHKW\\test_2k_y.csv', header=None, index=None)
#train_2k = pd.DataFrame(df2_fftlist_psd[0:2000], columns=np.arange(0,256,1))
#train_all = pd.DataFrame(df2_lmlist, columns=np.arange(0,256,1))



#for B082 (as list)
b082_orig_rsd = df4['RawStreamData'].values.tolist()

#hexlist is a list containing every hexstring split into 2 Bytes
df4_hexlist=[]
for elem in b082_orig_rsd:
    df4_hexlist.append(splitTo4(elem))

#for every record in column rawstreamdata
#create
# 1. another column with a list of hex values split into 4 characters
# 2. another list with the integer values

#intlist is a list containing the hexstring as an integer list
df4_intlist = []
#df2_mean_intlist= []
for ele1 in df4_hexlist:
    templist = []
    for ele2 in ele1:
        templist.append(h2i(ele2))
    df4_intlist.append(templist)

df4_hexlist.clear()

#df2['IntStreamData'] = df2_intlist
#df2_sl = df2['streamlength'].values.tolist()

"""df2_fftlist = []
for elex in df2_intlist:
    tlist = []
    for eley in elex:
        x = np.array(df2_intlist[eley])
        xf = fft(x)
        xf = xf.tolist()
        tlist.append(xf)
    df2_fftlist.append(tlist) delete after working"""


df4_fftlist = []
df4_intlist = np.array(df4_intlist)
for el in df4_intlist:
    temp = fft(el)
    temp = temp.tolist()
    df4_fftlist.append(temp)
    #tlist = []
    #tlist = np.empty([512])
    #for e in el:
        #x = np.array(df2_intlist[eley])
        #xf = fft(e)
        #xf = xf.tolist()
        #tlist.append(xf)
    #df2_fftlist.append(tlist)


df4_fftlist_half = []
for elem in df4_fftlist:
    len = elem.__len__()
    mid = len//2
    half_list = elem[:mid]
    df4_fftlist_half.append(half_list)

df4_fftlist_psd = []
for elem in df4_fftlist_half:
    tlist = []
    for ele in elem:
        ele = np.real(ele * np.conj(ele))
        tlist.append(ele)
    df4_fftlist_psd.append(tlist)

#save memory df3_fftlist.clear()
#save memory df3_fftlist_half.clear()


train_10k_z = pd.DataFrame(df4_fftlist_psd[0:10000], columns=np.arange(0,256,1))
train_10k_z.to_csv(wd + '\\Data\\BHKW\\train_10k_z.csv', header=None, index=None)
test_2k_z = pd.DataFrame(df4_fftlist_psd[10000:12000], columns=np.arange(0,256,1))
test_2k_z.to_csv(wd + '\\Data\\BHKW\\test_2k_z.csv', header=None, index=None)
#train_2k = pd.DataFrame(df2_fftlist_psd[0:2000], columns=np.arange(0,256,1))
#train_all = pd.DataFrame(df2_lmlist, columns=np.arange(0,256,1))



df2_time = df2.iloc[10000:12000]
df2_time = df2_time.drop(columns=['UUID','streamlength','RawStreamData'])
df2_time.to_csv(wd + '\\Data\\BHKW\\df2_time.csv', header=None)


"""
#x = np.array(df2_intlist[0])
#xf = fft(x)
#df2_fftlist.append(xf)

#x = np.array(df2_intlist[1])
#xf = fft(x)
#df2_fftlist.append(xf)

y = np.array(df2_intlist[0])    #Array of 1 stream values, consisting of 512 or 720 or values
yf = fft(y)     #Fourier transform of that particular 1 stream of values
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)    #Consider frequency only upto Nyquist


#df2_real_yf = complex2real(yf) #convert 528 complex numbers to real
#normalized_real_yf = normalize(real_yf) #normalize these real values between 0 and 1


"""

"""Calculating the mean of a given stream of list
#counter = 0
for ele in df2_intlist:
    if len(ele)==0:
        df2_mean_intlist.append(0)
        #print("{} list empty".format(counter))
    else:
        df2_mean_intlist.append(mean(ele))
    #counter+=1

df2['MeanIntList'] = df2_mean_intlist
df1_sl = df1['streamlength'].values.tolist()
"""

"""

#configure a loop to go over the entire list of data points
from scipy.fft import fft, ifft
import numpy as np
import matplotlib.pyplot as plt

fourier_list=[]
df2_xf_list = []
count_df2_sl = 0

for item in df2_sl:
    N = item
    T = 1/6664
    x = np.linspace(0,N*T,N)
    y = np.array(df2_intlist[count_df2_sl])
    yf = fft(y)
    fourier_list.append(yf)
    xf = np.linspace(0, 1 / (2 * T), N // 2)
    df2_xf_list.append(xf)
    count_df2_sl = count_df2_sl + 1
    if count_df2_sl>3:
        break
    plt.stem(xf, 2.0 / N * np.abs(yf[0:N // 2]), linefmt='-')
    plt.xlabel('Frequency (Hz)'), plt.ylabel('Amplitude')
    # plt.xticks(np.arange(min(xf), max(xf), 20))
    # plt.yticks(0,1)
    plt.xlim(0, 3332)
    plt.ylim(0, 20)
    plt.show()

#working model for one data point
from scipy.fft import fft, ifft
import numpy as np
import matplotlib.pyplot as plt

N = 512    #Number of datapoints in one stream
T = 1.0/6664     #frequency in Hz
x = np.array(df2_intlist[0])    #Array of 1 stream values, consisting of 512
xf = fft(x)     #Fourier transform of that particular 1 stream of values


#plot the above numbers
plt.stem(xf, 2.0/N * np.abs(yf[0:N//2]),linefmt='-')
plt.xlabel('Frequency (Hz)'), plt.ylabel('Amplitude')
#plt.xticks(np.arange(min(xf), max(xf), 20))
#plt.yticks(0,1)
plt.xlim(0, 3332)
plt.ylim(0,3)
plt.show()


xmin = 0
xmax = 3332
ymin = 0
ymax = 100

#plt.plot(xf, 2.0/N * np.abs(yf[0:N//2])) #plot only the magnitude (real value) and not the angle(phase)
#plt.axis([xmin,xmax,ymin,ymax])
#plt.grid()
#plt.show()































#for B081
b081_orig_rsd = df3['RawStreamData'].values.tolist()

#hexlist is a list containing every hexstring split into 2 Bytes
df3_hexlist=[]
for elem in b081_orig_rsd:
    df3_hexlist.append(splitTo4(elem))

#for every record in column rawstreamdata
#create
# 1. another column with a list of hex values split into 4 characters
# 2. another list with the integer values

#intlist is a list containing the hexstring as an integer list
df3_intlist = []
df3_mean_intlist= []
for ele1 in df3_hexlist:
    templist = []
    for ele2 in ele1:
        templist.append(h2i(ele2))
    df3_intlist.append(templist)

df3['IntStreamData'] = df3_intlist

#counter = 0
for ele in df3_intlist:
    if len(ele)==0:
        df3_mean_intlist.append(0)
        #print("{} list empty".format(counter))
    else:
        df3_mean_intlist.append(mean(ele))
    #counter+=1

df3['MeanIntList'] = df3_mean_intlist


#for B082
b082_orig_rsd = df4['RawStreamData'].values.tolist()

#hexlist is a list containing every hexstring split into 2 Bytes
df4_hexlist=[]
for elem in b082_orig_rsd:
    df4_hexlist.append(splitTo4(elem))

#for every record in column rawstreamdata
#create
# 1. another column with a list of hex values split into 4 characters
# 2. another list with the integer values

#intlist is a list containing the hexstring as an integer list
df4_intlist = []
df4_mean_intlist= []
for ele1 in df4_hexlist:
    templist = []
    for ele2 in ele1:
        templist.append(h2i(ele2))
    df4_intlist.append(templist)

df4['IntStreamData'] = df3_intlist

#counter = 0
for ele in df4_intlist:
    if len(ele)==0:
        df4_mean_intlist.append(0)
        #print("{} list empty".format(counter))
    else:
        df4_mean_intlist.append(mean(ele))
    #counter+=1

df4['MeanIntList'] = df4_mean_intlist"""