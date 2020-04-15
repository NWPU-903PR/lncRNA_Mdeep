# lncRNA_Mdeep
An alignment-free predictor for long non-coding RNAs identification by multimodal deep learning

lncRNA_Mdeep is a computational tool for distinguishing lncRNAs from protein-coding transcripts. It efficiently incorporates the hand-crafted features and raw sequence information by a multimodal deep learning framework to predict the probability whether a transcript is lncRNA or not. LncRNA_Mdeep achieves a good performance on human dataset and multiple cross-species datasets.

lncRNA_Mdeep is currently pre-trained on human training dataset and the pre-trained model is available in ./models/best_models.h5. 

## Dependencies: 
(1) numpy; (2) biopthon; (3) keras
To install the dependencies, type the following command:

    $ sudo pip3 install numpy
    $ sudo pip3 install biopthon
    $ sudo pip3 install keras

## Usage:
    $ python3 lncRNA_Mdeep.py -i (fasta_file) -o (output.txt)
  
   ## input.fasta: input file, it should be a FASTA sequence file, e.g.:

    $ python3 lncRNA_Mdeep.py -i (fasta_file) -o (output.txt)
    >transcripts _1
       GTGCACACGGCTCCCATGCGTTGTCTTCCGAGCGTCAGGCCGCCCCTACCCGTGCTTTCTGCTCTGCAGACCCTCTTCCTAGACCTCCGTCCTTTGTCCCATCGCTGCCTTCCCCTCAAGCTCAGGGCCAAGCTGTCCGCCAACCTCGGCTCCTCCGGGCAGCCCTCGCCCGGGGTGCGCCCCGGGGCAGGACCCCCAGCCCACGCCCAGGGCCCGCCCCTGCCCTCCAGCCCTACGCCTTGACCCGCTTTCCTGCGTCTCTCAGCCTACCTGACCTTGTCTTTACCTCTGT…
    >transcripts_2
    TCAGCCTCCCAAGTAGCTGGGGCTACAGGCACCTGCCACCAAACCCGGCTAATTTTTTTGTATTTTTAGTAGAGACGGGGTTTCACCGTGTTAGCCAGGATCGTCTTGATCTCCTGACCTTGTGATCCACCCGCCTCGGCCTCCCAAATTGCTGGGATTACAGATGTGAGCCACCGCACCTGGTCCAAGAACCCAAGTTTTAGATCTAGAGTGATGTCAGCATGACATTGATTTCCTGAGGCCCAGGGGCGAAGGAGCTGAGGACAGCAGAGGGGTG…
   ## output.txt: output file, it shows the predicted probability whether the inputs transcripts are lncRNA or not, e.g.:
    0.87
    0.21
    # If the predicted probability is bigger than 0.5, the corresponding input transcript is lncRNA. 
    Otherwise, the input transcript is not lncRNA. 

# Contact:
If you have any questions, please do not hesitate to contact us.
Shao-Wu Zhang, zhangsw@nwpu.edu.cn
Xiao-Nan Fan, fanxn@mail.nwpu.eud.cn
