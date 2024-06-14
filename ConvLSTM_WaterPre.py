# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 13:25:16 2023

@author: wisdom
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import math
# import os
# import pathlib

# 归一化函数
def NormData(data):
    max_data=np.amax(data,axis=0)
    max_data=2 * max_data
    min_data=np.amin(data,axis=0)
    min_data=0.5 * min_data
    max_min=max_data-min_data
    norm_data=np.zeros([data.shape[0],data.shape[1]])
    for i in range(data.shape[1]):
        norm_data[:,i]=(data[:,i]-min_data[i])/max_min[i]
    return norm_data

#反归一化函数
def No_NormData(data,nordata):
    max_data=np.amax(data,axis=0)
    min_data=np.amin(data,axis=0)
    no_norm_data=np.zeros([nordata.shape[0],nordata.shape[1]])
    for i in range(nordata.shape[1]):
        no_norm_data[:,i]=nordata[:,i]*(2*max_data[i]-0.5*min_data[i])+0.5*min_data[i]
    return no_norm_data
    
def CreatDataset(Waterdata, data, TimeSteps):
    # (1)数据重构：2维-->3维
    data=np.reshape(data,(data.shape[0],19,1)) # (*,19,1)
    
    # (2)训练集数据提取:2018-2022四年1826天数据
    train_size=1826-TimeSteps
    train_data = np.zeros((train_size, TimeSteps, 19, 1),dtype = float)
    for group in range(0,train_size):
        temp=data[group:group+TimeSteps,:,:] #TimeSteps天的水位为一组，
        train_data[group,:,:,:]=temp
    train_data = np.transpose(train_data, [0, 1, 3, 2])
    train_data = np.expand_dims(train_data, 4)
    train_x = train_data[:, :np.int(TimeSteps/2), :, :,:]
    train_y = train_data[:, np.int(TimeSteps/2):, :, :,:]
    
    # (3)测试集数据提取:2022年最后一组数据预测
    test_data = np.zeros((1, TimeSteps, 19, 1),dtype = float)
    test_data[0,:,:,:] =data[train_size:train_size+TimeSteps,:,:]
    test_data = np.transpose(test_data, [0, 1, 3, 2]) 
    test_data= np.expand_dims(test_data, 4)
    test_x = test_data[:, :np.int(TimeSteps/2), :, :,:]
    test_y = test_data[:, np.int(TimeSteps/2):, :, :,:]
    man_water=Waterdata[train_size+np.int(TimeSteps/2):train_size+TimeSteps,:]
    return train_x,train_y,test_x,test_y,man_water

def Build_model():
    filters = 15
    model = keras.Sequential(
        [
            keras.Input(
                shape=(None, 1, 19, 1)
            ),
            layers.ConvLSTM2D(
                filters=filters, kernel_size=(3, 2), padding="same", activation="tanh",return_sequences=True
            ),  #, dropout=0.02
            layers.BatchNormalization(),
            layers.ConvLSTM2D(
                filters=filters, kernel_size=(3, 2), padding="same", activation="tanh",return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.ConvLSTM2D(
                filters=filters, kernel_size=(3, 2), padding="same", activation="tanh",return_sequences=True
            ),
            #----一半输入、一半输出-------------------------------
            layers.BatchNormalization(),
            layers.ConvLSTM2D(
                filters=filters, kernel_size=(3, 2), padding="same", activation="tanh",return_sequences=True
            ),            
            layers.BatchNormalization(),
            layers.Conv3D(
                filters=1, kernel_size=(3, 2, 1), padding="same", activation="tanh"            
            ),
        ]
    )
    # loss=binary_crossentropy
    model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(0.0001)) #学习率
    model.summary()
    return model

def Train_modle(model, train_x, train_y):
    # 参数设置
    Epochs = 1200 # In practice, you would need about 80,000 epochs.
    Batch_size=18
    # 定义回调：这里定义了早期终止训练和调整学习率回调函数，因此无需担心无效训练时间增加问题。
    early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10000, verbose=0, mode='auto',baseline=0.0005, restore_best_weights=False)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=50)
    model.fit(
        train_x,
        train_y,
        batch_size=Batch_size,
        epochs=Epochs,
        callbacks=[early_stopping],
        verbose=2,
        validation_split=0.05, #训练集和验证集的拆分
    )
    # # 训练过程损失函数
    # losses_ConvLSTM = model.history.history['loss']
    # plt.figure(figsize=(10,4))
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.xticks(np.arange(0,Epochs,Epochs/50))
    # plt.plot(range(len(losses_ConvLSTM)),losses_ConvLSTM)
    
    # val_losses_ConvLSTM = model.history.history['val_loss']
    # plt.figure(figsize=(10,4))
    # plt.xlabel("Epochs")
    # plt.ylabel("Val_Loss")
    # plt.xticks(np.arange(0,Epochs,Epochs/50))
    # plt.plot(range(len(val_losses_ConvLSTM)),val_losses_ConvLSTM)
    # print(seq.output_shape) 
    return model
    
# NSE求解函数
def NSE_coe(y_true, y_pred):
    SD=np.sum((y_true - y_pred) ** 2)
    SL=np.sum((y_true - np.mean(y_true)) ** 2)
    Nse=1-SD/SL
    return Nse

if __name__=='__main__':
    # 数据加载
    WaterLevelData=np.load('WaterLevelAll_V3.npy')
    WaterLevelData=WaterLevelData.transpose() #数据转置
    
    # 数据归一化
    NormWaterData=NormData(WaterLevelData)
    # 相关变量定义
    # TimeSteps 时间步长；Times 训练次数； PreDays 
    PreDays=15  #预测天数
    RMSE=np.zeros((PreDays,19),dtype = float)
    MAE=np.zeros((PreDays,19),dtype = float)
    NSE=np.zeros((PreDays,19),dtype = float)
    # Pre_Water=np.zeros((PreDays*5,15),dtype = float)  #保存反归一化后的预测结果
    Flag=0
    
    # MSSTCN 模型构建
    for TimeSteps in range(30,32,2):
        # 数据集划分
        train_x,train_y,test_x,test_y,man_water=CreatDataset(WaterLevelData,NormWaterData,TimeSteps)
        #模型构建
        model = Build_model()
        #模型训练
        print('convSLTM training: TimeSteps={TimeSteps}'.format(TimeSteps=TimeSteps))
        model = Train_modle(model, train_x, train_y)
        for Times in range(0,1):
            #水位预测
            prediction = model.predict(test_x)
            # 预测结果反归一化
            pre_water=np.zeros((np.int(TimeSteps/2), 19),dtype = float)
            for days in range(0,np.int(TimeSteps/2)):
                pre_water[days,:]=prediction[0,days,:,:,0]
            pre_water=No_NormData(WaterLevelData,pre_water)
           
                      
           #误差计算：(1)计算19个站点每天的RMSE
            for stas in range(0,19):
                RMSE[Flag,stas]= math.sqrt(mean_squared_error(pre_water[:,stas], man_water[:,stas]))    
                MAE[Flag,stas]=mean_absolute_error(pre_water[:,stas], man_water[:,stas])
            #误差计算：(2)计算预测天中每个站点的NSE
            for days in range(0,np.int(TimeSteps/2)):
                RMSE[Flag+1,days]= math.sqrt(mean_squared_error(pre_water[days,:], man_water[days,:]))    
                MAE[Flag+1,days]=mean_absolute_error(pre_water[days,:], man_water[days,:])
                NSE[Flag+1,days]=NSE_coe(man_water[days,:],pre_water[days,:])
        Flag=Flag+1
                
        # 预测结果展示、对比: 各站点
        for stas in range(0,19):
            plt.figure(figsize=(6,4))
            plt.xlabel("Days",fontsize=10)
            plt.ylabel("Water level (m)",fontsize=10)
            plt.plot(man_water[:,stas],'b')
            plt.plot(pre_water[:,stas],'r')
            plt.legend(labels=['MeasuredValue','PredictedValue'],loc='best',fontsize=10)
            plt.xticks(np.arange(0,np.int(TimeSteps/2)))
            plt.title("Station: {stas},  RMSE={rmse:.3f}".format(stas=stas+1, rmse=RMSE[0,stas]),fontsize=10)
            plt.grid(True)