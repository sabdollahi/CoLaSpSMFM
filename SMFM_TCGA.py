#SMFM uses four different mutation interpreters and a matrix factorization method to predict Uncertain Significant genes
#Feedforward Neural Network model and Obtain the accuracy of cancer type prediction 
import pickle
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import seaborn as sns
sns.set(style="whitegrid")

#SMFM-based (Mutation Interpreters) Feedforward Neural Network Architecture
class SMFMFFNN(nn.Sequential):
    def __init__(self, x_dim):
        super(SMFMFFNN, self).__init__()
        self.linear1 = nn.Linear(x_dim, 128)
        self.bn1 = nn.BatchNorm1d(num_features = 128)
        self.relu1 =  nn.ReLU()
        self.d1 = nn.Dropout(0.05)
        self.linear2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(num_features = 256)
        self.relu2 =  nn.ReLU()
        self.d2 = nn.Dropout(0.05)
        self.linear3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(num_features = 128)
        self.relu3 = nn.ReLU()
        self.d3 = nn.Dropout(0.05)
        self.linear4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(num_features = 64)
        self.relu4 = nn.ReLU()
        self.d4 = nn.Dropout(0.05)
        self.linear5 = nn.Linear(64, 16)
        self.bn5 = nn.BatchNorm1d(num_features = 16)
        self.relu5 = nn.ReLU()
        self.d5 = nn.Dropout(0.05)
        self.linear6 = nn.Linear(16, 2)
         
    def forward(self,x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.d1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.d2(x)
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.d3(x)
        x = self.linear4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.d4(x)
        x = self.linear5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.d5(x)
        x = self.linear6(x)
        return x

main_folder = "SMFMevaluation/"
tcga_clinical_dataframe = pickle.load(open(main_folder + "TCGA_clinical_dataframe.pickle","rb"))
tcga_SMFM_dataframe = pickle.load(open(main_folder + "OncomineSMFM.pickle","rb"))
tcga_clinical_dataframe = tcga_clinical_dataframe[['cancer_type']]
cancer_organ_dataframe = pd.read_excel(main_folder + "cancer_type.xlsx")
cancer_organ_dataframe = cancer_organ_dataframe[['cancer_type','organ']]
organs_dataframe = tcga_clinical_dataframe.join(cancer_organ_dataframe.set_index('cancer_type'), on='cancer_type')
SMFM_cancerType_df = tcga_SMFM_dataframe.join(organs_dataframe)
tcga_clinical_dataframe = pickle.load(open(main_folder + "TCGA_clinical_dataframe.pickle","rb"))
x_dim = 195
classes = {"adrenal gland":0, "bladder":1, "breast":2, "GYN":3, "bile duct":4, "CRC":5, "bone marrow":6, "esophagus":7, "brain":8, "head and neck":9, "Kidney":10, "liver":11, "Lung":12, "pleura":13, "pancreatic":14, "male reproductive system":15, "other":16, "Melanoma":17, "stomach":18, "thyroid":19, "thymus":20}
which_clinicals = ['cancer_class']
tcga_clinical_dataframe = tcga_clinical_dataframe[which_clinicals]
dic_cancer_accuracies = {}
for cancer_class in classes:
    print("-------------" + cancer_class + "-------------")
    folds_accuracy = []
    replace_statement = {}
    for cl in classes:
        if(cl != cancer_class):
            replace_statement[cl] = 0
        else:
            replace_statement[cl] = 1
    specific_cancer_patients = tcga_clinical_dataframe[tcga_clinical_dataframe["cancer_class"] == cancer_class]
    specific_cancer_patients = specific_cancer_patients.replace({'cancer_class': replace_statement})
    other_cancer_patients = tcga_clinical_dataframe[tcga_clinical_dataframe["cancer_class"] != cancer_class]
    other_cancer_patients = shuffle(other_cancer_patients).sample(n = len(specific_cancer_patients))
    other_cancer_patients = other_cancer_patients.replace({'cancer_class': replace_statement})
    K = 10 #Kfold (number of parts = K)
    kf_other = KFold(n_splits = K, shuffle = True)
    kf_specific = KFold(n_splits = K, shuffle = True)
    parts_specific = kf_specific.split(specific_cancer_patients)
    parts_other = kf_other.split(other_cancer_patients)
    indices_specific = next(parts_specific, None)
    indices_other = next(parts_other, None)
    while(indices_specific):
        #Define the model
        model = SMFMFFNN(x_dim)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        batch_size = 20
        for shuffled_epoch in range(20):
            training = specific_cancer_patients.iloc[indices_specific[0]]
            training_other = other_cancer_patients.iloc[indices_other[0]]
            training = shuffle(training.append(training_other))
            Y = training[['cancer_class']].values
            Y = Variable(torch.LongTensor(Y.flatten()), requires_grad=False)
            training = training.index
            for epoch in range(50):
                for index in range(0, len(training), batch_size):
                    y = Y[index : index + batch_size]
                    batch = SMFM_cancerType_df.loc[training[index : index + batch_size]]
                    batch = batch.to_numpy()
                    X = batch[:,0:x_dim].astype(float)
                    X = torch.FloatTensor(X)
                    optimizer.zero_grad()
                    Y_hat = model(X)
                    loss = criterion(Y_hat, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
        test = specific_cancer_patients.iloc[indices_specific[1]]
        test_other = other_cancer_patients.iloc[indices_other[1]]
        test = shuffle(test.append(test_other))
        Y_test = test[['cancer_class']].values
        Y_test = Variable(torch.LongTensor(Y_test.flatten()), requires_grad=False)
        test = test.index
        avg_acc = 0
        ii = 0
        for index in range(0, len(test), batch_size):
            y = Y_test[index : index + batch_size]
            test_list = SMFM_cancerType_df.loc[test[index : index + batch_size]]
            test_list = test_list.to_numpy()
            test_list = test_list[:,0:x_dim].astype(float)
            test_list = torch.FloatTensor(test_list)
            if(len(test_list) <= 1):
                break
            test_batch_Y_hat = model.forward(test_list)
            dummy, preds_test = torch.max (test_batch_Y_hat, dim = 1)
            print(preds_test)
            accuracy_test = (preds_test == y).long().sum().float() /  preds_test.size()[0]
            avg_acc += accuracy_test
            ii += 1
        avg_acc = avg_acc / ii
        print("ACC: " + str(avg_acc))
        folds_accuracy.append(avg_acc)
        indices_specific = next(parts_specific, None)
        indices_other = next(parts_other, None)
    dic_cancer_accuracies[cancer_class] = folds_accuracy
pickle.dump(dic_cancer_accuracies, open(main_folder + "RESULTS/SMFM-CancerPrediction-Accuracies.pickle","wb"))                    
