import pandas as pd
import numpy as np
import random
from sklearn.metrics import f1_score
import MGD
import spectrogram
import mix_noise

#get onehot cpded output
def to_onehot(emotion,num_classes):
    out = np.zeros(num_classes,dtype=np.int8)
    if(isinstance(emotion, str)):
        out[int(emotion)-1]=1
    return out

def get_clip_len(clip):
    return clip.shape[1]

def get_sorted_df(df):
    clip_lens=df['data'].apply(func=get_clip_len,)
    df['spec_len']=clip_lens
    df=df.sort_values(by=['spec_len'],ignore_index=True)
    return df

def get_cropped_spec(spec,req_len):
    max_start=spec.shape[1]-req_len
    start=random.randint(0,max_start)
    cropped_spec=spec[:,start:(start+req_len)]
    return cropped_spec

def get_random_sample(data_list):
    l=len(data_list)
    index=random.randint(0,l-1)
    data=data_list[index][0]
    if(data.shape[-1]>3):
        data=np.expand_dims(data,axis=-1)
    data=np.expand_dims(data,axis=0)
    label=data_list[index][1]
    return data,label

def get_batch(comb_data,max_bs=32,threshold=0):
    data_list=[]
    label_list=[]
    len_list=[]
    for data in comb_data:
        data_list.append(data[0])
        label_list.append(data[1])
        len_list.append(data[0].shape[1])

    spec_df=pd.DataFrame(columns=['data','label','spec_len'])
    spec_df['data']=data_list
    spec_df['label']=label_list
    spec_df['spec_len']=len_list

    req_len=random.randint(120,600)
    
    spec_df['diff_len']=spec_df['spec_len']-req_len
    selected_df=spec_df[(spec_df['diff_len']) > threshold] 
    selected_df=selected_df.sample(frac=1).reset_index(drop=True)
    df_len=len(selected_df)
    if(df_len>max_bs):
        selected_df=selected_df[0:max_bs]
    cropped_data=selected_df['data'].apply(get_cropped_spec,args=[req_len])
    labels=selected_df['label']
    labels=labels.reset_index()

    data_list=[]
    for item in cropped_data:
        data_list.append(item)
    data_list=np.array(data_list)
    data_list=np.expand_dims(data_list,axis=3)
    
    labels_list=[]
    for item in labels['label']:
        labels_list.append(item)
    labels_list=np.array(labels_list)  
    labels_list=np.squeeze(labels_list)
    return data_list,labels_list

def print_shape_stats(spec_list):
    shape_list=[]
    for spec in spec_list:
        shape_list.append(spec[0].shape[1])
    shape_list=np.array(shape_list)
    print((np.min(shape_list),np.mean(shape_list),np.max(shape_list)))
    
def get_vali_acc(model,data_list):
    iscorrect_list=[]
    pred_list=[]
    true_list=[]
    is_correct_list=[]
    for (data,label) in data_list:
        data=np.expand_dims(data,axis=0)
        if(data.shape[-1]>3):
            data=np.expand_dims(data,axis=-1)
        pred=model.predict(data)
        pred=np.argmax(pred)
        iscorrect=(pred==np.argmax(label))
        iscorrect_list.append(iscorrect)  
        
        pred_list.append(pred)
        true_list.append(np.argmax(label))
        is_correct_list.append(iscorrect)
        
    iscorrect_list=np.array(iscorrect_list)
    iscorrect_list.astype(int)
    iscorrect_list=iscorrect_list.astype(int)
    f1=f1_score(true_list,pred_list,average=None)
    return np.sum(iscorrect_list)/len(iscorrect_list),np.mean(f1),f1

def get_len(item):
    return item.shape[0]

def get_padded(d,pad_len):
    if(pad_len>len(d)):
        padded=np.pad(d,pad_width=(max_len-len(d),0),mode='constant')
    else:
        padded=d[0:pad_len]
    return padded
def get_padded_df(df):
    lens=df['data'].apply(get_len).values
    pad_len=np.max(lens)
    padded=df['data'].apply(get_padded,args=[pad_len])
    df=df.drop(['data'],axis=1)
    df['data']=padded
    return df


def get_emo_num(ar):
    return np.argmax(ar)

def get_class_weights(data_df):
    labels=data_df['label'].apply(get_emo_num).values
    total=0
    num_classes=np.max(labels)+1
    for i in range(0,num_classes):
        val=labels[labels==i].shape[0]
        total+=val
    class_weights=[]
    for i in range(0,num_classes):
        val=labels[labels==i].shape[0]
        class_weights.append(1/(val/total))
    return class_weights

def get_fold(fold=0):
    data=pd.read_pickle('/scratch/lnw8px/emodb_audio/audio_data/random_data.pkl')
    vali_len=data.shape[0]*0.1
    start=vali_len*fold
    end=start+vali_len
    start=int(start)
    end=int(end)
    select=[False]*data.shape[0]
    select[start:end]=[True]*(end-start)
    data['select']=select

    train_df=data[data['select']==False].reset_index(drop=True)
    vali_df=data[data['select']==True].reset_index(drop=True)
    return train_df,vali_df

def get_noise_fold_data(fold):
    train_df,vali_df=get_fold(fold=fold)
    class_weights=get_class_weights(train_df)
    WN_train_df=mix_noise.get_white_noise_dataframe(train_df,SNR=40)
    RN_train_df=mix_noise.get_random_noise_dataframe(train_df,SNR=40)
    WN_RN_train_df=mix_noise.get_random_noise_dataframe(WN_train_df,SNR=40)
    train_data=train_df.append(WN_train_df).append(RN_train_df).append(WN_RN_train_df)
    return train_data,vali_df,class_weights

def get_noise2_fold_data(fold):
    train_df,vali_df=get_fold(fold=fold)
    class_weights=get_class_weights(train_df)
    WN_train_df=mix_noise.get_white_noise_dataframe(train_df,SNR=40)
    RN_train_df=mix_noise.get_random_noise_dataframe(train_df,SNR=40)
    train_data=train_df.append(WN_train_df).append(RN_train_df)
    return train_data,vali_df,class_weights

def get_Wnoise_fold_data(fold):
    train_df,vali_df=get_fold(fold=fold)
    class_weights=get_class_weights(train_df)
    WN_train_df=mix_noise.get_white_noise_dataframe(train_df,SNR=40)
    train_data=train_df.append(WN_train_df)
    return train_data,vali_df,class_weights

def add_random_noise(data):
    mul=random.randint(0,10)/100
    rand=np.random.rand(data.shape[0],data.shape[1],data.shape[2],data.shape[3])*mul
    return data+rand

def get_noise2_data(data_df):
    WN_train_df=mix_noise.get_white_noise_dataframe(data_df,SNR=40)
    RN_train_df=mix_noise.get_random_noise_dataframe(data_df,SNR=40)
    data_df=data_df.append(WN_train_df).append(RN_train_df)
    return data_df

#**************************************
#**************************************
#*******mag*************************
def get_mag_fold(fold,fft_len,fft_hop,height):
    train_df,vali_df=get_fold(fold=fold)
    class_weights=get_class_weights(train_df)

    spectrogram.set_param(fft_len=fft_len,fft_hop=fft_hop,img_height=height,sample_len=2000)   
    train_list=spectrogram.get_spectro_data_list(train_df)
    vali_list=spectrogram.get_spectro_data_list(vali_df)
    return train_list,vali_list,class_weights

def get_mag_data_list(data_df,fft_len,fft_hop,height):
    spectrogram.set_param(fft_len,fft_hop,height,2000)   
    data_list=spectrogram.get_spectro_data_list(data_df)
    return data_list

#**************************************
#**************************************
#*******phase*************************
def get_phase_fold(fold):
    train_df,vali_df=get_fold(fold=fold)
    class_weights=get_class_weights(train_df)

    spectrogram.set_param(fft_len=fft_len,fft_hop=fft_hop,img_height=height,sample_len=2000)   
    train_list=spectrogram.get_phase_data_list(train_df)
    vali_list=spectrogram.get_phase_data_list(vali_df)
    return train_list,vali_list,class_weights

def get_phase_data_list(data_df,fft_len,fft_hop,height):
    spectrogram.set_param(fft_len=fft_len,fft_hop=fft_hop,img_height=height,sample_len=2000)   
    data_list=spectrogram.get_phase_data_list(data_df)
    return data_list


#**************************************
#**************************************
#*******MGD*************************
def get_MGD_fold_features(fold,fft_len,fft_hop,height):
    train_df,vali_df=get_fold(fold=fold)
    class_weights=get_class_weights(train_df)
    MGD.set_param(fft_len=fft_len,fft_hop=fft_hop,img_height=height,sample_len=2000)   
    train_list=MGD.get_data_list(train_df)
    vali_list=MGD.get_data_list(vali_df)
    return train_list,vali_list,class_weights

def get_MGD_data_list(data_df,fft_len,fft_hop,height):
    MGD.set_param(fft_len=fft_len,fft_hop=fft_hop,img_height=height,sample_len=2000)   
    data_list_MGD=MGD.get_data_list(data_df)
    return data_list_MGD



#**************************************
#**************************************
#*******comb2*************************
def get_comb2_data(data1_list,data2_list):
    data_list=[]
    for i in range(0,len(data1_list)):
        data1=data1_list[i]
        data2=data2_list[i]
        label=data1[1]
        comb=np.append(data1[0],data2[0],axis=1)
        data=[comb,label]
        data_list.append(data)
    return data_list

def get_comb2_data_list(data_df,fft_len,fft_hop,height):
    #get combined data
    MGD.set_param(fft_len=fft_len,fft_hop=fft_hop,img_height=height,sample_len=2000)   
    data_list_MGD=MGD.get_data_list(data_df)

    spectrogram.set_param(fft_len=fft_len,fft_hop=fft_hop,img_height=height,sample_len=2000)   
    data_list_mag=spectrogram.get_spectro_data_list(data_df)

    data_list=get_comb2_data(data_list_mag,data_list_MGD)
    return data_list

def get_comb2_fold_features(fold,fft_len,fft_hop,height):
    train_df,vali_df=get_fold(fold=fold)
    class_weights=get_class_weights(train_df)
    train_list=get_comb2_data_list(train_df,fft_len,fft_hop,height)
    vali_list=get_comb2_data_list(vali_df,fft_len,fft_hop,height)
    return train_list,vali_list,class_weights

#**************************************
#**************************************
#*******comb1*************************
def get_comb1_data(data1_list,data2_list):
    data_list=[]
    for i in range(0,len(data1_list)):
        data1=data1_list[i]
        data2=data2_list[i]

        label=data1[1]
        data1_ar=data1[0]
        data2_ar=data2[0]

        data1_ar=np.expand_dims(data1_ar,axis=-1)
        data2_ar=np.expand_dims(data2_ar,axis=-1)

        if(not (data1_ar.shape==data2_ar.shape)):
            min_len=min(data1_ar.shape[1],data2_ar.shape[1])
            if(data1_ar.shape[1]>data2_ar.shape[1]):
                data1_ar=data1_ar[:,0:min_len,:]
            else:
                data2_ar=data2_ar[:,0:min_len,:]            

        comb=np.append(data2_ar,data1_ar,axis=2)
        data=[comb,label]
        data_list.append(data)
    return data_list


def get_comb1_data_list(data_df,fft_len,fft_hop,height):
    #get combined data
    MGD.set_param(fft_len=fft_len,fft_hop=fft_hop,img_height=height,sample_len=2000)   
    data_list_MGD=MGD.get_data_list(data_df)

    spectrogram.set_param(fft_len=fft_len,fft_hop=fft_hop,img_height=height,sample_len=2000)   
    data_list_mag=spectrogram.get_spectro_data_list(data_df)

    data_list=get_comb1_data(data_list_mag,data_list_MGD)
    return data_list

def get_comb1_fold_features(fold,fft_len,fft_hop,height):
    train_df,vali_df=get_fold(fold=fold)
    class_weights=get_class_weights(train_df)
    train_list=get_comb1_data_features(train_df,fft_len,fft_hop,height)
    vali_list=get_comb1_data_features(vali_df,fft_len,fft_hop,height)
    return train_list,vali_list,class_weights

'''
def get_comb1_data_list(data_df):
    #get combined data
    MGD.set_param(fft_len=fft_len,fft_hop=fft_hop,img_height=height,sample_len=2000)   
    data_list_MGD=MGD.get_data_list(data_df)

    spectrogram.set_param(fft_len=fft_len,fft_hop=fft_hop,img_height=height,sample_len=2000)   
    data_list_mag=spectrogram.get_spectro_data_list(data_df)

    data_list=get_comb1_data(data_list_mag,data_list_MGD)
    return data_list
'''

    