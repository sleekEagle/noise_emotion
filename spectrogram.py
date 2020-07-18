from scipy import signal
import pandas as pd
from random import randint
import numpy as np
import librosa


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
    
def get_spec_time_freq(spec_len):
    #calculate frequency range of the specrtoograms
    max_cycles_per_frame=int(n_fft/2)
    max_freq=max_cycles_per_frame/(n_fft/sr)
    max_freq_spec=max_freq/max_cycles_per_frame*height
    #calculate the time duration of the spectrograms
    spec_time=(spec_len-1)*hop_length/sr+n_fft/sr
    return max_freq_spec,spec_time

emotions=np.array(['W','L','E','A','F','T','N'])
#helper function to calculate emotion number from np array
def get_emo_num(ar):
    num=np.where(ar[0]==1)[1][0]
    return num

#mel spectrogram in dB values (hence log)
sr=22050
'''
def get_log_mel_spectrogram(voice):
    #scale voice to [0,1]
    voice=np.interp(voice, (voice.min(), voice.max()), (0, 1))
    
    f,t,sxx=signal.spectrogram(voice,fs=sr,window='hamming',nperseg=n_fft,noverlap=n_fft-hop_length,nfft=n_fft,mode='magnitude')
    #get power spectrum
    pow_frames=(sxx**2)
    
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (f[-1]) / 700))  # Convert Hz to Mel
    
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_ceps + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((n_fft + 1) * hz_points / sr)
    
    fbank = np.zeros((num_ceps, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, num_ceps + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames.T, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = (20 * np.log10(filter_banks)).T  # dB
    filter_banks=(filter_banks-np.min(filter_banks))/(np.max(filter_banks)-np.min(filter_banks))
    return filter_banks
'''

def get_log_mel_spectrogram(voice):
    voice=np.interp(voice, (voice.min(), voice.max()), (0, 1))
    S = librosa.feature.melspectrogram(voice, sr=sr, n_fft=n_fft, 
                                           hop_length=hop_length, 
                                           n_mels=num_ceps)
    S_DB = librosa.power_to_db(S, ref=np.max)
    S_DB=(S_DB-np.min(S_DB))/(np.max(S_DB)-np.min(S_DB))
    return S_DB

def get_phase_spectrogram(voice):
    #scale voice to [0,1]
    voice=np.interp(voice, (voice.min(), voice.max()), (0, 1))
    
    f,t,sxx=signal.spectrogram(voice,fs=sr,window='hamming',nperseg=n_fft,noverlap=n_fft-hop_length,nfft=n_fft,mode='phase')
    spec=(sxx-np.min(sxx))/(np.max(sxx)-np.min(sxx))
    return spec

def get_log_spectrogram(voice):
    #scale voice to [0,1]
    voice=np.interp(voice, (voice.min(), voice.max()), (0, 1))
    voice=voice-np.mean(voice)
    f,t,sxx=signal.spectrogram(voice,fs=sr,window='hamming',nperseg=n_fft,noverlap=n_fft-hop_length,nfft=n_fft,mode='magnitude')
    spec = np.log(sxx+0.00001)
    spec=spec[0:height,:]
    spec=(spec-np.min(spec))/(np.max(spec)-np.min(spec))
    return spec

def print_stats(array):
    print("mean = " + str(np.mean(array)) + " std = " + str(np.std(array)) + " min = " + str(np.min(array)) + " max = " + str(np.max(array))) 
    
def zero_mean(signal):
    return signal-np.mean(signal)

def get_zero_mean_df(df):
    norm=df['data'].apply(zero_mean)
    norm_df=pd.DataFrame(df['label'])
    norm_df['data']=norm
    return norm_df

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

def get_spectro_data(df):
    spec=df['data'].apply(get_log_spectrogram)
    labels=df['label'].values
    labels=np.expand_dims(labels,axis=1)
    labels=np.apply_along_axis(get_emo_num,axis=1,arr=labels)
    data_list=[]
    for item in list(spec):
        data_list.append(item)
    return np.array(data_list),labels

def get_spectro_data_list(df):
    spec=df['data'].apply(get_log_spectrogram)
    labels=df['label'].values
    labels=np.expand_dims(labels,axis=1)
    data_list=[]
    for i,item in enumerate(list(spec)):
        data_list.append((item,labels[i][0]))
    return data_list

def get_phase_data_list(df):
    spec=df['data'].apply(get_phase_spectrogram)
    labels=df['label'].values
    labels=np.expand_dims(labels,axis=1)
    data_list=[]
    for i,item in enumerate(list(spec)):
        data_list.append((item,labels[i][0]))
    return np.array(data_list)

#length is length in seconds 
def crop_audio(signal,length=2):
    #length of classification window in seconds
    sample_len=length
    frame_len=int(sample_len*sr)
    max_start=len(signal)-frame_len
    if(max_start<0):
        cropped_signal=signal
        diff=-max_start
        cropped_signal=np.pad(cropped_signal,pad_width=(0,diff),mode='constant',constant_values=0)
    else:
        start=randint(0,max_start)
        cropped_signal=signal[start:start+frame_len]
    return cropped_signal

def get_cropped_df(df,length):
    cropped_df=df['data'].apply(crop_audio,length=length)
    df=pd.DataFrame(df['label'])
    df['data']=cropped_df
    return df

def get_shape(row):
    return row.shape[0]
#gives min, mean and max length of audio data in the dataframe in seconds
def get_audio_df_stats(df):
    shapes=df['data'].apply(get_shape)
    return np.min(shapes.values)/sr,np.mean(shapes.values)/sr,np.max(shapes.values)/sr