from train import *
from dataPre import *

import pandas as pd

def main():

    train = train_kfold(trainArgs)
    result = train.kfold()
    result.to_csv(trainArgs['filename'], mode='w')

    idx = result.groupby(['Fold'])['Valid_AUC'].transform(max) == result['Valid_AUC']
    result = result[idx]
    print('****************************************************************************')
    print(">>[Test Result] avg.auc : {:.4f}, f1 : {:.4f}\n"
            .format(result['Test_AUC'].mean(), result['Test_F1'].mean()))

if __name__ == "__main__":
    main()