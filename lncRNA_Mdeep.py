# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 23:15:32 2019

@author: xiaonan 
"""

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
import argparse
from model import processing
import keras
from keras.models import load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.models import Model, Sequential
from keras.layers import Conv2D, GlobalMaxPool2D
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn import metrics

def cnn_model(x_train, y_train):
    cnn = Sequential()
    cnn.add(Conv2D(filters=64,
                   kernel_size=(4,200),
                   input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
                   strides=(1, 1)))
    cnn.add(BatchNormalization(epsilon=1e-06, momentum=0.9))
    cnn.add(Activation('relu'))
    cnn.add(GlobalMaxPool2D(name='cnn_feas'))

    cnn.add(Dropout(0.3))
    cnn.add(Dense(1, activation='sigmoid'))

    cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    cnn.fit(x_train, y_train,
            epochs=20,
            batch_size=64,
            verbose=2)
    return cnn

def sae_model(x_train, y_train):
    autoencoder = Sequential()
    autoencoder.add(Dense(128, activation='relu', name='encoder_layer0', input_dim=x_train.shape[1]))
    autoencoder.add(Dropout(0.5))
    autoencoder.add(Dense(64, activation='relu', name='encoder_layer1'))
    autoencoder.add(Dropout(0.5))
    autoencoder.add(Dense(32, activation='relu', name='encoder_layer2'))

    autoencoder.add(Dropout(0.5))
    autoencoder.add(Dense(1, activation='sigmoid'))
    autoencoder.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    autoencoder.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    verbose=2)
    return autoencoder

def dnn(x_train, y_train):
    dnn = Sequential()
    dnn.add(Dense(4, activation='relu', name='dnn_seq_layer1', input_dim=x_train.shape[1]))
    dnn.add(BatchNormalization(name='dnn_seq_br1'))
    dnn.add(Dense(4, activation='relu', name='dnn_seq_layer2'))
    dnn.add(BatchNormalization(name='dnn_seq_br2'))

    dnn.add(Dropout(0.5))
    dnn.add(Dense(1, activation='sigmoid'))

    dnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    dnn.fit(x_train, y_train,
            epochs=10,
            batch_size=128,
            verbose=2)
    return dnn


def merge_model(x_train_dnn, x_train_cnn, x_train_sae, y_train):
    autoencoder = sae_model(x_train_sae, y_train)
    encoder_model = Model(inputs=autoencoder.input,
                          outputs=autoencoder.get_layer('encoder_layer2').output)
    dnn_model = dnn(x_train_dnn, y_train)
    dnn_fea_model = Model(inputs=dnn_model.input,
                          outputs=dnn_model.get_layer('dnn_seq_br2').output)
    cnn = cnn_model(x_train_cnn, y_train)
    cnn_fea_model = Model(inputs=cnn.input,
                          outputs=cnn.get_layer('cnn_feas').output)

    concat_layer0 = keras.layers.concatenate([dnn_fea_model.output, cnn_fea_model.output, encoder_model.output], axis=1)
    concat_layer0 = BatchNormalization(name='concat_br1')(concat_layer0)
    concat_layer0 = Dropout(0.5, name='dr1')(concat_layer0)
    concat_layer0 = Dense(16, activation='relu')(concat_layer0)
    concat_layer0 = BatchNormalization(name='concat_br2')(concat_layer0)

    concat_layer0 = Dropout(0.5, name='dr2')(concat_layer0)
    concat_layer3 = Dense(1, activation='sigmoid')(concat_layer0)

    merge_model = Model(inputs=[dnn_fea_model.input, cnn_fea_model.input, encoder_model.input],
                        outputs=concat_layer3)
    merge_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    merge_model.fit([x_train_dnn, x_train_cnn, x_train_sae], y_train,
                    epochs=10,
                    batch_size=128,
                    shuffle=1,
                    verbose=2)
    return merge_model



def extract_feas(fastafile):
    transcripts = SeqIO.parse(fastafile, "fasta")
    k=6
    maxlen = 3000
    ofh_feas = np.empty(shape=[0, 4])
    kmer_feas = np.empty(shape=[0, 4**k])
    onehot_feas = np.empty(shape=[0, 4, maxlen])
    seq_num = 0
    seq_id = []
    for seq_recode in transcripts:
        seq_id.append(seq_recode.id)
        seq = seq_recode.seq
        seq = seq.back_transcribe()
        seq_num += 1
        ofh, kmer, onehot = processing.extract_feas(seq, k, maxlen)
        ofh_feas = np.concatenate((ofh_feas, ofh), axis=0)
        kmer_feas = np.concatenate((kmer_feas, kmer), axis=0)
        dim = onehot_feas.shape
        onehot_feas = np.append(onehot_feas, onehot)
        onehot_feas = onehot_feas.reshape(dim[0]+1,dim[1],dim[2])
    onehot_feas = np.reshape(onehot_feas, (onehot_feas.shape[0], onehot_feas.shape[1], onehot_feas.shape[2], 1))
    return ofh_feas, kmer_feas, onehot_feas, seq_id

def lncRNA_mdeep_retrain(dataset, y_train):
    y_train = np.loadtxt(y_train, dtype=int)
    train_data_ofh, train_data_kmer, train_data_onehot, seq_id = extract_feas(dataset)
    model = merge_model(train_data_ofh, train_data_onehot, train_data_kmer, y_train)
    model.save('./model/new_model.h5')


    
def lncRNA_mdeep(inputdata, output):
    merge_model = load_model('./model/best_model.h5')
    data_ofh, data_kmer, data_onehot, seq_id = extract_feas(inputdata)
    p_test = merge_model.predict([data_ofh, data_onehot, data_kmer])
    
    writefile = open(output,'w')
    numseq = len(p_test)
    for i in range(numseq):
        if p_test[i] > 0.5:
            labels = 'noncoding'
        else:
            labels = 'coding'
        writefile.write(seq_id[i]+'\t'+str(labels)+'\t'+str(p_test[i])+'\n')
    writefile.close()
    
    
parser = argparse.ArgumentParser(description="lncRNA_Mdeep: an alignment-free predictor for long non-coding RNAs identification by multimodal deep learning")
parser.add_argument('-retrain', type=str, help='training dataset for retraining a new model')
parser.add_argument('-l', type=str, help='label for training data')
parser.add_argument('-i', type=str, help='input fasta')
parser.add_argument('-o', type=str, help='output file')

args = parser.parse_args()
dataset = args.retrain
if dataset is None:
    inputdata = args.i
    if inputdata is None:
        print('please provide the input fasta file')
    else:
        output = args.o
        if output is None:
            output = "output.txt"
        lncRNA_mdeep(inputdata, output)
else:
    y_train = args.l
    if y_train is None:
        print('If you are trying to retrain a new model, please provide the label of your training data')
    else:
        print('Start to retrain a new model')
        lncRNA_mdeep_retrain(dataset, y_train)




