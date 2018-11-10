# TASK

The task is to predict mnist digits, while trying to minimize the false positive rates by using a decision mechanism that says 'not sure' for data points its not sure.

![alt text](CF.png)
Confusion Matrix

The basic model is a cnn based upon [1]. It consists of 3 blocks of convolutions, each block having the same number of filters. Summarized as 

784 - [32C3-32C3-32C5S2] - DRPO - [64C3-64C3-64C5S2] - DRPO - 128F -  DRPO - 10F - sigmoid

32C3 := Filters = 32 , kernel size = 3, stride = 1
32C3 := Filters = 32 , kernel size = 3, stride = 1
32C3 := Filters = 32 , kernel size = 5, stride = 2

64C3 := Filters = 64 , kernel size = 3, stride = 1
64C3 := Filters = 64 , kernel size = 3, stride = 1
64C3 := Filters = 64 , kernel size = 5, stride = 2

128F := Fully Connected Layer with 128 as output
10F := Fully Connected Layer with 10 as output

DRPO := Dropout with probability to drop as .4
Each layer other than the final layer are relu
Each layer has batch-normalization 

This leads to the following scores.

Class | True Pos. | True Neg. | False Pos. | False Neg. |
------|---------|---------|--------|--------|
   0  | 99.8    | 99.99   | 0.01    | 0.2  |
   1  | 99.7    | 99.97   | 0.03    | 0.3  |
   2  | 99.5    | 99.91   | 0.09    | 0.5  |
   3  | 99.2    | 99.98   | 0.02    | 0.8  |
   4  | 99.2    | 99.87   | 0.13    | 0.8  |
   5  | 99.9    | 99.95   | 0.04    | 0.1  |
   6  | 99.7    | 99.93   | 0.07    | 0.3  |
   7  | 99.4    | 99.85   | 0.14    | 0.6  |
   8  | 99.2    | 99.91   | 0.09    | 0.7  |
   9  | 99.9    | 99.92   | 0.08    | 2.1  |
 Mean | 99.36   | 99.93   | 0.07    | 0.64 |


The second part of the task is to reduce the false negative rate, by not predicting some of the data points based on a criteria. I use bayesian CNN for this part as proposed in [2]. In this work Gal et al. proposed that dropouts can be interpretted as an ensemble of several models while testing whereas each configuration being one model while training. This leads to the fact that the prediction of the model at test time is an aggregation of a distribution over models, hence the variance in prediction (note with dropouts at test times too) can be treated as model variance.

I use this variance as a sign of the model not being sure on the input with a threshold of .04

After using the decision criteria, the results are the following

Class | True Pos. | True Neg. | False Pos. | False Neg. |
------|---------|---------|--------|--------|
  0   | 100.0   | 99.9886 | 0.01   | 0.0    |
  2   | 99.89   | 100.0   | 0.0    | 0.0    |
  1   | 100.0   | 100.0   | 0.0    | 0.10   |
  3   | 99.89   | 100.0   | 0.0    | 0.10   |
  4   | 99.78   | 99.96   | 0.035  | 0.21   |
  5   | 99.88   | 100.0   | 0.035  | 0.11   |
  6   | 99.78   | 99.98   | 0.0    | 0.21   |
  7   | 100.0   | 99.98   | 0.023  | 0.0    |
  8   | 99.78   | 99.98   | 0.023  | 0.21   |
  9   | 99.67   | 99.98   | 0.022  | 0.33   |
 Mean | 99.87   | 99.985  | 0.014  | 0.12   |

Coverage Percentage : 93.09%

To run the code, follow these steps:

* Save the dataset in train.csv in the same directory

```bash
python helperMNIST.py [--val 0/1] # creates test, validation (if val=1), train set from .csv file and saves in data folder
```
 
```bash
python main.py [--tr] [--n_epochs] [--logInt] [--momentum] [--lr] [--batchSize] [--dataPath] [--msp] [--vt] [--ns] [--lm] [--pt] [--vl] [--m] [--d]
```

Parameters:
* tr : 0 means use decision-making to get results. 1 (default) means train normal model
* msp : path to saved model
* vt : threshold for decision-making function
* ns : number of samples for decision-making
* pt : threshold for probabilities of sigmoid
* lm : load last train model of corresponding network
* m : network number. Default is 2 which gives best performance
* vl : to use validation set or not. Default 1
* d : dropout value
* others are self-explanatory, not to be changed generally 

Example 


```ash
python main # train a model with default paramters
```

```bash
python main --ns=500 --t=0 --vt=.014 # use large value of ns only on GPU, will be slow on cpu
```

[1] https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist

[2] http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf

<!-- 
# DATA DESCRIPTION

The data file mnist.csv contains gray-scale images of hand-drawn digits,
from zero through nine.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784
pixels in total. Each pixel has a single pixel-value associated with it,
indicating the lightness or darkness of that pixel, with higher numbers meaning
darker. This pixel-value is an integer between 0 and 255, inclusive.

The data set (mnist.csv), has 785 columns. The first column, called
"label", is the digit that was drawn by the user. The rest of the columns
contain the pixel-values of the associated image.

Each pixel column in the training set has a name like pixelx, where x is an
integer between 0 and 783, inclusive. To locate this pixel on the image,
suppose that we have decomposed x as x = i * 28 + j, where i and j are integers
between 0 and 27, inclusive. Then pixelx is located on row i and column j of a
28 x 28 matrix, (indexing by zero).

For example, pixel31 indicates the pixel that is in the fourth column from the
left, and the second row from the top, as in the ascii-diagram below.

Visually, if we omit the "pixel" prefix, the pixels make up the image like this:

000 001 002 003 ... 026 027
028 029 030 031 ... 054 055
056 057 058 059 ... 082 083
 |   |   |   |  ...  |   |
728 729 730 731 ... 754 755
756 757 758 759 ... 782 783 

# ACKNOWLEDGEMENTS
More details about the dataset, including algorithms that
have been tried on it and their levels of success, can be found at
http://yann.lecun.com/exdb/mnist/index.html. The dataset is made available
under a Creative Commons Attribution-Share Alike 3.0 license. -->
