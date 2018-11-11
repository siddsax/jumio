# TASK

The task is to predict mnist digits, while trying to minimize the false positive rates by using a decision mechanism that says 'not sure' for data points its not sure.

![alt text](CF.png)
**Confusion Matrix**

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

Class | True Pos. | False Pos. |
------|---------|---------|
   0  | 99.8    | 0.055  |
   1  | 99.7    | 0.011  |
   2  | 99.6    | 0.066  |
   3  | 99.4    | 0.044  |
   4  | 99.6    | 0.122  |
   5  | 99.3    | 0.055  |
   6  | 99.7    | 0.067  |
   7  | 99.4    | 0.055  |
   8  | 99.5    | 0.067  |
   9  | 99.5    | 0.067  |
 **Mean** | **99.45**  | **0.061** |

The second part of the task is to reduce the false negative rate, by not predicting some of the data points based on a criteria. I employed two techniques to achieve this.

a) **Bayesian CNN** as proposed in [2]. In this work Gal et al. proposed that dropouts can be interpretted as an ensemble of several models while testing whereas each configuration being one model while training. This leads to the fact that the prediction of the model at test time is an aggregation of a distribution over models, hence the variance in prediction (note with dropouts at test times too) can be treated as model variance. I use this variance as a sign of the model not being sure on the input with a threshold of .01 (set by validation)

b) **Low probabilities** If the model predicted an output with probability less than .975 (set by validation), then as well I  discarded that as the model being not sure as this means the model is giving high probability to another class as well.

After using the decision criteria, the results are the following

Class | True Pos. | False Pos. |
------|---------|---------|
  0   | 100.0   | 0.012  |
  2   | 100.0   | 0.0   |
  1   | 100.0   | 0.0   |
  3   | 100.0   | 0.0   |
  4   | 100.0   | 0.012 |
  5   | 99.88   | 0.0 |
  6   | 100.0   | 0.012   |
  7   | 100.0   | 0.0 |
  8   | 100.0   | 0.0 |
  9   | 99.77   | 0.0 |
 **Mean** | **99.96**   | **0.003**  |

**Coverage Percentage : 91.18%**

To run the code, follow these steps:

* Save the dataset in train.csv in the same directory

```bash
python helperMNIST.py [--val 0/1] # creates test, validation (if val=1), train set from .csv file and saves in data folder
```
 
```bash
python main.py [--tr] [--n_epochs] [--logInt] [--momentum] [--lr] [--batchSize] [--dataPath] [--msp] [--vt] [--ns] [--lm] [--pt] [--vl] [--m] [--d]
```

The models and data splits can be downloaded from [here](http://cse.iitk.ac.in/users/siddsax/jumio.zip) 

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


```bash
python main # train a model with default paramters
```

```bash
python main --ns=500 --t=0 --vt=.014 # use large value of ns only on GPU, will be slow on cpu
```

[1] https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist

[2] http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf
