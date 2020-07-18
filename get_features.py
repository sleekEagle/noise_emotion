import spectrogram
import MGD
import numpy as np
import helper

emotions=np.array(['W','L','E','A','F','T','N'])

def to_onehot(num):
    num_classes=emotions.shape[0]
    out = np.empty([0,num_classes])
    for x in np.nditer(num):
        onehot = np.zeros(num_classes)
        onehot[int(x)] = 1
        out = np.append(out,[onehot],axis = 0)
    return out

def get_mag_phase_data(train_df,vali_df,fft_len,fft_hop,height):
    spectrogram.set_param(fft_len=fft_len,fft_hop=fft_hop,sample_len=2000)    
    train_data_mag,train_labels=spectrogram.get_spectro_data(train_df)
    vali_data_mag,vali_labels=spectrogram.get_spectro_data(vali_df)

    train_data_phase,train_labels=spectrogram.get_phase_data(train_df)
    vali_data_phase,vali_labels=spectrogram.get_phase_data(vali_df)

    train_data_mag=np.expand_dims(train_data_mag,axis=1)
    train_data_phase=np.expand_dims(train_data_phase,axis=1)
    vali_data_mag=np.expand_dims(vali_data_mag,axis=1)
    vali_data_phase=np.expand_dims(vali_data_phase,axis=1)

    train_data_mag=train_data_mag[:,:,0:height,:]
    vali_data_mag=vali_data_mag[:,:,0:height,:]
    train_data_phase=train_data_phase[:,:,0:height,:]
    vali_data_phase=vali_data_phase[:,:,0:height,:]

    train_data_comb=np.append(train_data_mag,train_data_phase,axis=1)
    vali_data_comb=np.append(vali_data_mag,vali_data_phase,axis=1)
    return train_data_comb,vali_data_comb,train_labels,vali_labels

alpha=0.6
gamma=0.4
def get_mag_mgd_data(data_df,fft_len,fft_hop,height):
    spectrogram.set_param(fft_len=fft_len,fft_hop=fft_hop,sample_len=2000)    
    data_mag,labels=spectrogram.get_spectro_data(data_df)

    MGD.set_param(fft_len=fft_len,fft_hop=fft_hop,sample_len=2000)
    data_mgd,labels=MGD.get_data(data_df,gamma,alpha)

    data_mag=np.expand_dims(data_mag,axis=1)
    data_mgd=np.expand_dims(data_mgd,axis=1)

    data_mag=data_mag[:,:,0:height,:]
    data_mgd=data_mgd[:,:,0:height,:]

    data_comb=np.append(data_mag,data_mgd,axis=1)
    return data_comb,labels

def get_mgd_phase_data(data_df,fft_len,fft_hop,height):
    spectrogram.set_param(fft_len=fft_len,fft_hop=fft_hop,sample_len=2000)    
    data_phase,labels=spectrogram.get_phase_data(data_df)
    
    MGD.set_param(fft_len=fft_len,fft_hop=fft_hop,sample_len=2000)
    data_mgd,labels=MGD.get_data(data_df,gamma,alpha)

    data_phase=np.expand_dims(data_phase,axis=1)
    data_mgd=np.expand_dims(data_mgd,axis=1)
    data_phase=data_phase[:,:,0:height,:]
    data_mgd=data_mgd[:,:,0:height,:]

    data_comb=np.append(data_phase,data_mgd,axis=1)
    return data_comb,labels


def get_onehot_labels(labels):
    labels=np.expand_dims(labels,axis=1)
    labels=np.apply_along_axis(to_onehot,axis=1,arr=labels)
    labels=np.squeeze(labels)
    return labels


def get_autokeras_comb2_data(data_df,fft_len,fft_hop,height):
    #get combined data
    MGD.set_param(fft_len=fft_len,fft_hop=fft_hop,img_height=height,sample_len=2000)   
    data_list_MGD=MGD.get_data_list(data_df)

    spectrogram.set_param(fft_len=fft_len,fft_hop=fft_hop,img_height=height,sample_len=2000)   
    data_list_mag=spectrogram.get_spectro_data_list(data_df)

    data_list=helper.get_comb2_data(data_list_mag,data_list_MGD)
    
    data=[]
    labels=[]
    for d in data_list:
        data.append(d[0])
        labels.append(np.argmax(d[1]))
        
    return np.array(data),np.array(labels)