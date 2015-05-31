# ds_pcnns

Research on Distant Supervision for Relation Extraction via Piece-wise Convolutional Neural Networks

PCNNs+MIL implementation Python (theano) code.

Using DS dataset that was developed by Riedel et al. (2010).

Execute: python nyt_ds.py -e 13 -s 0 -u 230_27 -b 50 -w 3 -c tanh -d 50 -i gap_40_len_80 -n 0

# Abstract

Two problems arise when using distant supervision for relation extraction. First, in this method, an already existing knowledge base is heuristically aligned to texts, and the alignment results are treated as labeled data. 
However, the heuristic alignment can fail, resulting in wrong label problem. In addition, in previous approaches, statistical models have typically been applied to ad hoc features. The noise that originates from the feature extraction process can cause poor performance.

In this paper, we propose a novel model dubbed the Piecewise Convolutional Neural Networks (PCNNs) with multi-instance learning to address these two problems. 
To solve the first problem, distant supervised relation extraction is treated as a multi-instance problem in which the uncertainty of instance labels is taken into account. 
To address the latter problem, we avoid feature engineering and instead adopt convolutional neural networks with piecewise max pooling to automatically learn relevant features.
Experiments show that our method is effective and outperforms several competitive baseline methods. 
