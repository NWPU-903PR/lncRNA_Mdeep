# lncRNA_Mdeep
An alignment-free predictor for long non-coding RNAs identification by multimodal deep learning

lncRNA_Mdeep is a computational tool for distinguishing lncRNAs from protein-coding transcripts. It efficiently incorporates the hand-crafted features and raw sequence information by a multimodal deep learning framework to predict the probability whether a transcript is lncRNA or not. LncRNA_Mdeep achieves a good performance on human dataset and multiple cross-species datasets.

## Prerequisites: 
(1) numpy; (2) biopython; (3) keras

To install numpy, biopython, and keras, type the following command:

    $ sudo pip3 install numpy
    $ sudo pip3 install biopython
    $ sudo pip3 install keras

## Usage:
### 1）identify lncRNAs

    $ python3 lncRNA_Mdeep.py -i (fasta_file) -o (output.txt)
  
   ##### input.fasta: input file, it should be a FASTA sequence file, e.g.:
    >transcripts _1
    GTGCACACGGCTCCCATGCGTTGTCTTCCGAGCGTCAGGCCGCCCCTACCCGTGCTTTCTGCTCTGCAGACCCTCTTCCTAGACCTCCGTCCTTTGTCCCATCGCTGCCTTCCCCTCAAGCTCAGGGCCAAGCTGTCCGCCAACCTCGGCTCCTCCGGGCAGCCCTCGCCCGGGGTGCGCCCCGGGGCAGGACCCCCAGCCCACGCCCAGGGCCCGCCCCTGCCCTCCAGCCCTACGCCTTGACCCGCTTTCCTGCGTCTCTCAGCCTACCTGACCTTGTCTTTACCTCTGT…
    >transcripts_2
    TCAGCCTCCCAAGTAGCTGGGGCTACAGGCACCTGCCACCAAACCCGGCTAATTTTTTTGTATTTTTAGTAGAGACGGGGTTTCACCGTGTTAGCCAGGATCGTCTTGATCTCCTGACCTTGTGATCCACCCGCCTCGGCCTCCCAAATTGCTGGGATTACAGATGTGAGCCACCGCACCTGGTCCAAGAACCCAAGTTTTAGATCTAGAGTGATGTCAGCATGACATTGATTTCCTGAGGCCCAGGGGCGAAGGAGCTGAGGACAGCAGAGGGGTG…
   ##### output.txt: output file. The first column shows the transcript id, the second column shows the predicted label, and the third column shows the predicted probability, e.g.:
    transcripts _1 noncoding [0.87]
    transcripts _2 coding [0.21]

### 2）retrain a new model

       $ python3 lncRNA_Mdeep.py -retrain (training.fasta) -l (labels.txt)
         
   ##### training.fasta: The training transcripts for retraining a new model. 
   ##### labels.txt. It should be a table with the binary value in one column, representing the coresponding labels for training data, e.g.:
    1
    1
    0
    0
    1
    # If the label = 1, the coresponding transcript in training data is a lncRNA. 
    # If the label = 0, the coresponding transcript in training data is a protein-coding RNA. 
    # Please make sure the number of labels is equal to the the number of training transcripts.
       
     ## note: Currently we cannot adjust the hyper-parameters automatically. 
     ## The re-trained new model will be named as 'new_model.h5' and saved in the fold of [model]

# Contact:
If you have any questions, please do not hesitate to contact us.

Shao-Wu Zhang, zhangsw@nwpu.edu.cn

Xiao-Nan Fan, fanxn@mail.nwpu.eud.cn
