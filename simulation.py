
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import warnings

warnings.filterwarnings('ignore')




'''
Remember in TP, FP, TN and FN
first letter represents, 'What is my model's prediction? t/f'
second letter represents, 'what did my model predict it to be? p/n'
so it is always about the model.

FP is 'TYPE1' error, that is the predicted value is False, that is model thinks it is positive when it is actually negative.
FN is 'TYPE2' error, that is the predicted value is False, that is model thinks it is negative when it is actually positive.


Graph metric
ROC curve is a plot of TPR vs FPR:
TPR = TP/(TP + FN) => Recall/sensitivity
FPR = FP/(FP + TN) => 1 - Specificity
TP + FN = all actual positive samples in the dataset
TN + FP = all actual negative samples in the dataset
The way we get a curve here is because, we change the threshold, which is where the FP, FN TP and TN tend to change.
Animation reference : https://paulvanderlaken.com/2019/08/16/roc-auc-precision-and-recall-visually-explained/


Value metric
Accuracy = (TP+TN)/(TP+TN+FP+FN)
Precision = TP/(TP+FP), which means, it could also be TN/(TN+FN), how many of the positive samples according to the model were actually right? and same for the negative samples.
Recall = TP/(TP+FN), which means, it could also be TN/(TN+FP), how many of the actual positive/ negative samples is the model able to predict. How well is the model able to recall the positive samples or negative samples, from all the actual positive/negative samples.
F1 Score = 2/((1/precision)+(1/recall))which is the HM of precision and recall

'''
MEAN_POS = 1
MEAN_NEG = 1

STD_DEV_POS = 1
STD_DEV_NEG = 1

NO_POINTS_POS = 10
NO_POINTS_NEG = 100

# value of threshold can only be in between 0 and 1.
THRESHOLD = 0.5

def gen_dataset(mean_pos, mean_neg, std_pos, std_neg, no_pos, no_neg):
    positive_datasamples = np.random.normal(loc=mean_pos, scale=std_pos, size=no_pos)
    negative_datasamples = np.random.normal(loc=mean_neg, scale=std_neg, size=no_neg)

    df = pd.DataFrame(columns=['Samples', 'Actual_Class', 'Predicted_Class'])
    df['Predicted_Class'] = None
    for i in positive_datasamples:
        df = df.append({'Samples':i, 'Actual_Class':1}, ignore_index=True)
    for i in negative_datasamples:
        df = df.append({'Samples':i, 'Actual_Class':0}, ignore_index=True)

    outlier_values = [max(positive_datasamples), min(positive_datasamples), max(negative_datasamples), min(negative_datasamples)]
    min_val = min(outlier_values)
    max_val = max(outlier_values)
    return df, min_val, max_val


def compute_predicted_class(df, threshold, pred_class='Predicted_Class', samples='Samples'):
    df.loc[df[samples]<threshold, pred_class] = 0.0
    df.loc[df[samples]>threshold, pred_class] = 1.0
    return df

def compute_tptnfpfn(df, act_class='Actual_Class', pred_class='Predicted_Class', sample='Samples'):
    tp = df.loc[(df[act_class] == 1) & (df[pred_class] == 1)][sample].count()
    tn = df.loc[(df[act_class] == 0) & (df[pred_class] == 0)][sample].count()
    fp = df.loc[(df[act_class] == 0) & (df[pred_class] == 1)][sample].count()
    fn = df.loc[(df[act_class] == 1) & (df[pred_class] == 0)][sample].count()
    return tp, tn, fp, fn

def create_confusion_matrix(df):
    tp, tn, fp, fn = compute_tptnfpfn(df)
    # need to create a seperate animation side showing all these values
    #   
def tpr_fxn(tp, tn, fp, fn):
    tpr = float(tp)/float((tp+fn)+0.0001)
    return tpr
def fpr_fxn(tp, tn, fp, fn):
    fpr = float(fp)/float((fp+tn)+0.0001)
    return fpr
def accuracy_fxn(tp, tn, fp, fn):
    accuracy = float((tp+tn))/float((tp+tn+fp+fn))
    return accuracy
def precision_fxn(tp, tn, fp, fn):
    precision = float(tp)/float((tp+fp)+0.0001)
    return precision
def recall_fxn(tp, tn, fp, fn):
    recall = float(tp)/float((tp+fn)+0.0001)
    return recall
def specificity_fxn(tp, tn, fp, fn):
    specificity = 1.0-float(fp)/float((fp+tn)+0.0001)
    return specificity
    
def tpr_fpr_arr(df, sample='Samples', pred_class='Predicted_Class'):
    TPR_arr = []
    FPR_arr = []
    threshold_lis = np.linspace(min_val, max_val, 100)[1:-1]
    for threshold in threshold_lis:
        df.loc[df['Samples']<threshold, 'Predicted_Class'] = 0.0
        df.loc[df['Samples']>threshold, 'Predicted_Class'] = 1.0
        tp, tn, fp, fn = compute_tptnfpfn(df)
        tpr = tpr_fxn(tp, tn, fp, fn)
        fpr = fpr_fxn(tp, tn, fp, fn)
        TPR_arr.append(tpr)
        FPR_arr.append(fpr)
    return TPR_arr, FPR_arr


# Graph representing the data
df, min_val, max_val = gen_dataset(MEAN_POS, MEAN_NEG, STD_DEV_POS, STD_DEV_NEG, NO_POINTS_POS, NO_POINTS_NEG)
compute_predicted_class(df)
TPR_arr, FPR_arr = tpr_fpr_arr(df)

# sns.kdeplot(df, x='Samples', hue='Actual_Class')
# Based on the threshold value, determine the value of the Predicted_Class in the dataframe

# Graph metric

'''
plt.plot(FPR_arr, TPR_arr)
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()
'''
