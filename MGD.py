from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import math
from scipy.fftpack import dct,idct
from scipy.signal import medfilt
from random import randint

#sampling rate of wav files check this before running this code!
sr=22050
#number of cep coefficients
num_ceps=40
def set_param(fft_len,fft_hop,img_height,sample_len):
    global n_fft,hop_length,height,frame_len
    n_fft = int(sr*fft_len*0.001)
    hop_length = int(sr*fft_hop*0.001)  
    frame_len=int(sample_len/fft_hop)
    height=img_height

emotions=np.array(['W','L','E','A','F','T','N'])
#helper function to calculate emotion number from np array
def get_emo_num(ar):
    num=np.where(ar[0]==1)[1][0]
    return num
def to_onehot(num):
    num_classes=emotions.shape[0]
    out = np.empty([0,num_classes])
    for x in np.nditer(num):
        onehot = np.zeros(num_classes)
        onehot[int(x)] = 1
        out = np.append(out,[onehot],axis = 0)
    return out

def scale_array(ar):
    return (ar-np.min(ar))/(np.max(ar)-np.min(ar))

def print_stats(array):
    print("mean = " + str(np.mean(array)) + " std = " + str(np.std(array)) + " min = " + str(np.min(array)) + " max = " + str(np.max(array))) 
    
def zero_mean(signal):
    return signal-np.mean(signal)

def get_zero_mean_df(df):
    norm=df['data'].apply(zero_mean)
    norm_df=pd.DataFrame(df['label'])
    norm_df['data']=norm
    return norm_df
    
#***convert complex np array to polar arrays (2 apprays; abs and angle)
def to_polar(complex_ar):
    return np.abs(complex_ar),np.angle(complex_ar)

def to_complex(abs_ar,angle_ar):
    real_ar=abs_ar*np.cos(angle_ar)
    img_ar=abs_ar*np.sin(angle_ar)
    complex_ar=real_ar+1j*img_ar
    return complex_ar

def cepstral_smooth(X):
    X_abs,X_angle=to_polar(X)
    #X_abs=np.log(X_abs+1)
    dctX=dct(X_abs)
    dctX_chop=dctX[:30]
    smoothedX=idct(dctX_chop,n=X_abs.shape[0])
    smoothedX=np.interp(smoothedX, (smoothedX.min(), smoothedX.max()), (np.min(X_abs), np.max(X_abs)))
    return smoothedX

#calculate MGD of a single frame
def get_mgd_feature_frame(x,gamma=0.5,alpha=0.6): 
    #moving avg filter
    #N=2
    #x=np.convolve(x, np.ones((N,))/N, mode='valid')
    n=np.arange(x.shape[0])
    nx=x*n
    X=np.fft.rfft(x)
    Y=np.fft.rfft(nx)
    S=cepstral_smooth(x)
    exp_S=np.exp(S)
    
    T=(X.real*Y.real+X.imag*Y.imag)/(np.power(exp_S,2*gamma))
    
    Tm=T/(np.abs(T))*np.power(np.abs(T),alpha)
    Tm=Tm/np.max(np.abs(Tm))
    Tm=np.nan_to_num(Tm)
    cep=dct(Tm)
    cep=cep[1:num_ceps+1]
    return cep

#calculate MGD of a single frame
def get_mgd_frame(x,gamma=0.9,alpha=1.8): 
    n=np.arange(x.shape[0])
    nx=x*n
    X=np.fft.rfft(x)
    X=X[1:]
    Y=np.fft.rfft(nx)
    Y=Y[1:]
    S=cepstral_smooth(X)

    T=(X.real*Y.real+X.imag*Y.imag)/(np.power(S,2*gamma))
    
    Tm=T/(np.abs(T))*np.power(np.abs(T),alpha)
    #Tm=Tm/np.max(np.abs(Tm))
    Tm=(Tm-np.min(Tm))/(np.max(Tm)-np.min(Tm))
    #Tm=np.log(Tm+0.01)
    return Tm

#get mgd spectrogram from a time series array
def get_mgd_spec(x,gamma,alpha):
    tm_list=[]
    for i in range(0,x.shape[0]-n_fft,hop_length):
        sample_window=x[i:i+n_fft]
        window=np.hamming(n_fft)
        hamming_window=sample_window*window
        tm=get_mgd_frame(hamming_window,gamma,alpha)
        tm_list.append(tm)
    
    TM=np.transpose(np.array(tm_list))
    TM=TM[0:height,:]
    TM=np.nan_to_num(TM)
    scaled=(TM-np.min(TM))/(np.max(TM)-np.min(TM))
    return scaled
 
def get_cropped_data(ar_list,frame_len):
    data=[]
    for ar in ar_list:
        diff=frame_len-ar.shape[1]
        crop_ar=ar
        if(ar.shape[1] < frame_len):
            crop_ar=np.pad(ar,pad_width=((0,0),(0,diff)),mode='mean')
        else:
            max_index=-diff
            start=randint(0,max_index)
            crop_ar=crop_ar[:,start:(start+frame_len)]
        data.append(crop_ar)
    #data=np.array(data)
    return data

def get_data(df,gamma=0.5,alpha=0.6):
    spec_list=[]
    for index,row in df.iterrows():
        data=row['data']
        #N=5
        #data=np.convolve(data, np.ones((N,))/N, mode='valid')
        data=np.interp(data, (data.min(), data.max()), (0, 1))
        spec=get_mgd_spec(data,gamma,alpha)
        spec_list.append(spec)
    '''
    croped_data=get_cropped_data(spec_list,frame_len)
    croped_data=np.array(croped_data)
    '''
    labels=df['label'].values
    labels=np.expand_dims(labels,axis=1)
    labels=np.apply_along_axis(get_emo_num,axis=1,arr=labels)
    return np.array(spec_list),labels

def get_data_list(df,gamma=0.4,alpha=0.6):
    spec=df['data'].apply(get_mgd_spec,args=[gamma,alpha])
    labels=df['label'].values
    labels=np.expand_dims(labels,axis=1)
    data_list=[]
    for i,item in enumerate(list(spec)):
        data_list.append((item,labels[i][0]))
    return data_list