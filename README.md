# Neural-Network

A short note on implementing an OCR based neural network in C++
INTRODUCTION
The idea of this project was to compare the effectiveness of implementing a neural network (NN) in C++ against one implemented in Matlab.
The program in Matlab itself is a simple application of NNs designed to read human written characters, i.e. optical character recognition 
(OCR). 
The challenge was to come up with an implementation of such a program in C++ with performance comparable to the one implemented in Matlab.

CHALLENGES FACED
It was first decided that the program had to be written using an object oriented approach and used a class labelled ‘Matrix’ to constitute
all the relevant matrix operations that are required in any standard NN; More than 25 such operations were needed to be defined.
It was later found however that a static array (Matrix) size was unable to accommodate enough elements, so a dynamic array using the
keyword ‘new’ was used. Although this solved the problem of limited array size, it introduced an even harder to solve problem of stack 
overflow. The majority of time spent on the project was put into solving this. After many attempts, it was found that using the dynamic 
array creation as little as possible while using pass by reference wherever possible eliminated this problem almost entirely.
Soon, a working NN was developed. However, the runtime of this algorithm was found to be more than 10 times slower than the one 
implemented in Matlab; it was also discovered that the root cause of this lay in the implementation of matrix multiplication which was
significantly slower in C++. This was mostly due to Matlab being extremely efficient at dealing with matrices, specifically matrix
multiplication. Thus an attempt was made at optimizing the matrix multiplication algorithm using Strassen’s matrix multiplication 
algorithm which took the remaining bulk of the time (still being worked on). A successful implementation of it that reduced the runtime
significantly with was soon met with the issue stack overflow; i.e. the program kept ending up asking for ever more memory which caused
it to crash. So, instead of using dynamic arrays, a vector implementation was developed. (Still being worked on)
