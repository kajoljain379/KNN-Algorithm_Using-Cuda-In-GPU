# KNN-Algorithm_Using-Cuda-In-GPU
This project contain serial implementation of KNN algorithm and compare its performance with parallel implementation in GPU.K-Nearest Neighbor (k-NN) is a classification algorithm
used in machine learning and data mining applications such as
email spam filtering, content retrieval , customer segmentation
in online shopping websites etc.In this algorithm, the variable
k is defined by the user and this classifiers find the k number for
training set, which has similar or is closest to the test data.The
k-NN algorithm can be used for classification, based on the
distance to the k nearest members ,the algorithm decides which
class the given input should belongs to and then give corresponding output.


## Objective
High parallelism can be achieved using GPU and in com-
paratively lesser cost than CPU.
So we are trying to do the same task in GPU which speedup
the execution by 70% .We did the implementation consists
of different level of parallelism like searching ,sorting and
other parallely executable task.Apart from other flexibility is also added so that it works on any number of train and test data.
