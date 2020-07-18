import pandas as pd
import math
import numpy as np
from random import randint


#SNR in dB
#given a signal and desired SNR, this gives the required AWGN what should be added to the signal to get the desired SNR
def get_white_noise(signal,SNR) :
    signal=np.interp(signal, (signal.min(), signal.max()), (-1, 1))
    #RMS value of signal
    RMS_s=math.sqrt(np.mean(signal**2))
    #RMS values of noise
    RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
    #Additive white gausian noise. Thereore mean=0
    #Because sample length is large (typically > 40000)
    #we can use the population formula for standard daviation.
    #because mean=0 STD=RMS
    STD_n=RMS_n
    noise=np.random.normal(0, STD_n, signal.shape[0])
    noise_added=noise+signal
    return noise_added


#given a signal, noise (audio) and desired SNR, this gives the noise (scaled version of noise input) that gives the desired SNR
def get_noise_from_sound(signal,noise,SNR):
    RMS_s=math.sqrt(np.mean(signal**2))
    #required RMS of noise
    RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
    
    #current RMS of noise
    RMS_n_current=math.sqrt(np.mean(noise**2))
    noise=noise*(RMS_n/RMS_n_current)
    
    return noise

def get_real_noise_mixed(signal,noise,SNR):
    #if noise is shorter in time than signal, repeat the noise
    #crop noise if its longer than signal
    while(len(noise)<len(signal)):
        noise=np.append(noise,noise)

    if(len(noise)>len(signal)):
        noise=noise[0:len(signal)]
        
    #scale both noises from -1 to +1
    signal=np.interp(signal, (signal.min(), signal.max()), (-1, 1))
    noise=np.interp(noise, (noise.min(), noise.max()), (-1, 1))
    
    scaled_noise=get_noise_from_sound(signal,noise,SNR=SNR)
    signal_noise=signal+scaled_noise
    return signal_noise

#add noise to rqandom frequencies 
def get_noise_of_length(signal,SNR):
    num_noise_freqs=3
    num_values=int(len(signal)/2)
    thickness=int(num_values*0.1)
    values=np.zeros(num_values)
    for i in range(0,num_noise_freqs):
            rand=randint(0, (len(values)-1))
            values[rand]=1
    values + values*0j
    noise=np.fft.irfft(values)
    noise_signal=get_real_noise_mixed(signal,noise,SNR=SNR)
    return noise_signal

#add random noise of random magnitude to all the frequencies
def get_random_white_noise(signal,SNR):
    num_values=len(signal)
    num_values=int(num_values/2)
    values=np.random.randint(3,size=num_values)
    values=values/np.max(values)
    noise=np.fft.irfft(values)
    pad_width=len(signal)-len(noise)
    noise=np.pad(noise,pad_width=((0,pad_width)),mode='mean')
    noise_signal=get_real_noise_mixed(signal,noise,SNR=SNR)
    return noise_signal

def get_random_noise_dataframe(df,SNR):
    if(SNR>10000):
        print("no noise added..")
        return df
    noise=df['data'].apply(get_noise_of_length,args=[SNR])
    noise_df=pd.DataFrame(df['label'])
    noise_df['data']=noise
    return noise_df


def get_white_noise_dataframe(df,SNR):
    if(SNR>10000):
        print("no noise added..")
        return df
    noise=df['data'].apply(get_white_noise,args=[SNR])
    noise_df=pd.DataFrame(df['label'])
    noise_df['data']=noise
    return noise_df

def get_real_noise_dataframe(df,noise,SNR):
    if(SNR>10000):
        print("no noise added..")
        return df
    noise=df['data'].apply(get_real_noise_mixed,args=[noise,SNR])
    noise_df=pd.DataFrame(df['label'])
    noise_df['data']=noise
    return noise_df

def add_silence(data):
    data=np.pad(data,pad_width=(int(data.shape[0]/4),int(data.shape[0]/4)),mode='constant',constant_values=(0,0))
    return data

def add_silence_df(df):
    data=df['data'].apply(add_silence)
    silence_df=pd.DataFrame(df['label'])
    silence_df['data']=data
    return silence_df