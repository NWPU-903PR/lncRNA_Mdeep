# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:51:19 2019

@author: xiaonan
"""
import numpy as np
import math

def get_k_alpha(k):
    k_alpha = []
    chars = ['A', 'C', 'G', 'T']
    base = len(chars)
    end = len(chars)**k
    for i in range(end):
        n = i
        chn = chars[int(n%base)]
        for j in range(1, k):
            n = n/base
            chn = chn + chars[int(n%base)]
        k_alpha.append(chn)
    return k_alpha

def get_kmer(seq, k):
    k_alpha = get_k_alpha(k)
    kmer_fnum = []
    seq_len = len(seq)
    for val in k_alpha:
        num = seq.count(val)
        if seq_len - 3 == 0:
            kmer_fnum.append(0)
        else:
            kmer_fnum.append(float(num)/(seq_len-3))
    kmer_fnum = np.matrix(kmer_fnum)
    return kmer_fnum    

def extract_longest_orf(seq):
    winner = 0
    result = (0,0,0)
    start_coden=['ATG']
    stop_coden=['TAG','TAA','TGA']
    for frame in range(3):
        start = frame
        codon = []
        while start + 3 <= len(seq):
            codon.append([seq[start:start+3], start])
            start += 3
        codon_gen = iter(codon)
        while True:
            try:
                tmp, index = next(codon_gen)
            except StopIteration:
                break 
            if tmp in start_coden or not start_coden and tmp not in stop_coden:
                orf_start = index  
                end = False
                while True:
                    try:
                        tmp, index = next(codon_gen)
                    except StopIteration:
                        end = True
                    if tmp in stop_coden:
                        end = True
                    if end:
                        orf_end = index + 3
                        L = (orf_end - orf_start)
                        if L > winner:
                            winner = L
                            result = (orf_start, orf_end, L)
                        if L == winner and orf_start < result[0]:
                            result = (orf_start, orf_end, L)
                        break  
    return (result[2], seq[result[0]:result[1]])

# Fickett TESTCODE data
# NAR 10(17) 5303-531
position_prob ={
'A':[0.94,0.68,0.84,0.93,0.58,0.68,0.45,0.34,0.20,0.22],
'C':[0.80,0.70,0.70,0.81,0.66,0.48,0.51,0.33,0.30,0.23],
'G':[0.90,0.88,0.74,0.64,0.53,0.48,0.27,0.16,0.08,0.08],
'T':[0.97,0.97,0.91,0.68,0.69,0.44,0.54,0.20,0.09,0.09]
}
position_weight={'A':0.26,'C':0.18,'G':0.31,'T':0.33}
position_para  =[1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1,0.0]

content_prob={
'A':[0.28,0.49,0.44,0.55,0.62,0.49,0.67,0.65,0.81,0.21],
'C':[0.82,0.64,0.51,0.64,0.59,0.59,0.43,0.44,0.39,0.31],
'G':[0.40,0.54,0.47,0.64,0.64,0.73,0.41,0.41,0.33,0.29],
'T':[0.28,0.24,0.39,0.40,0.55,0.75,0.56,0.69,0.51,0.58]
}
content_weight={'A':0.11,'C':0.12,'G':0.15,'T':0.14}
content_para  =[0.33,0.31,0.29,0.27,0.25,0.23,0.21,0.17,0]

def look_up_position_prob(value, base):
	if float(value)<0:
		return None
	for idx,val in enumerate (position_para):
		if (float(value) >= val):
			return float(position_prob[base][idx]) * float(position_weight[base])

def look_up_content_prob(value, base):
	if float(value)<0:
		return None
	for idx,val in enumerate (content_para):
		if (float(value) >= val):
			return float(content_prob[base][idx]) * float(content_weight[base])

def calculate_fickett_value(seq):
    if len(seq)<2:
        return 0
    fickett_score=0
    seq=seq.upper()
    total_base = len(seq)
    A_content = float(seq.count('A'))/total_base
    C_content = float(seq.count('C'))/total_base
    G_content = float(seq.count('G'))/total_base
    T_content = float(seq.count('T'))/total_base
    
    phase_0 = [seq[i] for i in range(0,len(seq)) if i % 3==0]
    phase_1 = [seq[i] for i in range(0,len(seq)) if i % 3==1]
    phase_2 = [seq[i] for i in range(0,len(seq)) if i % 3==2]
    
    A_position=max(phase_0.count('A'),phase_1.count('A'),phase_2.count('A'))/(min(phase_0.count('A'),phase_1.count('A'),phase_2.count('A')) +1.0)
    C_position=max(phase_0.count('C'),phase_1.count('C'),phase_2.count('C'))/(min(phase_0.count('C'),phase_1.count('C'),phase_2.count('C')) +1.0)
    G_position=max(phase_0.count('G'),phase_1.count('G'),phase_2.count('G'))/(min(phase_0.count('G'),phase_1.count('G'),phase_2.count('G')) +1.0)
    T_position=max(phase_0.count('T'),phase_1.count('T'),phase_2.count('T'))/(min(phase_0.count('T'),phase_1.count('T'),phase_2.count('T')) +1.0)
    
    fickett_score += look_up_content_prob(A_content,'A')
    fickett_score += look_up_content_prob(C_content,'C')
    fickett_score += look_up_content_prob(G_content,'G')
    fickett_score += look_up_content_prob(T_content,'T')
    fickett_score += look_up_position_prob(A_position,'A')
    fickett_score += look_up_position_prob(C_position,'C')
    fickett_score += look_up_position_prob(G_position,'G')
    fickett_score += look_up_position_prob(T_position,'T')
    return fickett_score


def word_generator(seq,word_size,step_size,frame=0):
	for i in range(frame,len(seq),step_size):
		word =  seq[i:i+word_size]
		if len(word) == word_size:
			yield word
            
def calculate_hexamer_value(seq,word_size,step_size,coding,noncoding):
    if len(seq) < word_size:
        return 0
    sum_of_log_ratio_0 = 0.0
    frame0_count=0.0
    for k in word_generator(seq=seq, word_size = word_size, step_size=step_size,frame=0):
        if (not k in coding) or (not k in noncoding):
            continue
        if coding[k]>0 and noncoding[k] >0:
            sum_of_log_ratio_0  +=  math.log( coding[k] / noncoding[k])
        elif coding[k]>0 and noncoding[k] == 0:
            sum_of_log_ratio_0 += 1
        elif coding[k] == 0 and noncoding[k] == 0:
            continue
        elif coding[k] == 0 and noncoding[k] >0 :
            sum_of_log_ratio_0 -= 1
        else:
            continue
        frame0_count += 1
    try:
        return sum_of_log_ratio_0/frame0_count
    except:
        return -1

def read_hexamer():
    coding={}
    noncoding={}
    for line in open('./model/human_hexamer'):
        line = line.strip()
        fields = line.split()
        if fields[0] == 'hexamer':
            continue
        coding[fields[0]] = float(fields[1])
        noncoding[fields[0]] =  float(fields[2])
    return coding, noncoding

def get_onehot(seq, maxlen):
    chars = ['A', 'C', 'G', 'T']
    seq_len = len(seq)
    onehot_fea = np.zeros(shape=[4, seq_len])
    for i, val in enumerate(seq):
        if val not in chars:
            onehot_fea[:,i] = np.array([0]*4)
        else:
            index = chars.index(val)
            onehot_fea[index, i] = 1
    if(seq_len >= maxlen):
        onehot_fea = onehot_fea[:,0:maxlen]
    else:
        onehot_fea = np.concatenate([onehot_fea, np.zeros([4, (maxlen - seq_len)])], axis = 1)
    return onehot_fea

def extract_feas(seq, k, maxlen):
    cdslen, cds_seq = extract_longest_orf(seq)
    fickett_score = calculate_fickett_value(cds_seq)
    coding,noncoding = read_hexamer()
    hexamer_score = calculate_hexamer_value(cds_seq,6,3,coding,noncoding)        
    ofh = np.matrix((cdslen, cdslen/len(seq), fickett_score, hexamer_score))
    kmer = get_kmer(cds_seq, k)
    onehot = get_onehot(seq, maxlen)
    return ofh, kmer, onehot
    