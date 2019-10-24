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
    
def lncRNA_mdeep(dataset, output):
    merge_model = load_model('./model/best_model.h5')
    data_ofh, data_kmer, data_onehot, seq_id = extract_feas(dataset)
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
parser.add_argument('-i', type=str, help='input fasta')
parser.add_argument('-o', type=str, help='output file')

args = parser.parse_args()
dataset = args.i
if dataset is None:
    print('please provide the input fasta file')
else:
    output = args.o
    if output is None:
        output = "output.txt"
    lncRNA_mdeep(dataset, output)

