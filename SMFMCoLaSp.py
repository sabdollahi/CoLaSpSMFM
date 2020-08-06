import pickle
import pandas as pd
import pyexcel as pe
import matplotlib.pyplot as plt
import csv
from os import listdir
from sklearn.model_selection import train_test_split
import keras
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
from sklearn.metrics import mean_absolute_error
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
import itertools
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
from skmultilearn.adapt import MLkNN
from sklearn.model_selection import GridSearchCV
from skmultilearn.adapt import BRkNNaClassifier
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import seaborn as sns


'''
    Creates DataFrame of Clinical Data from 'ExcelFiles' folder
    xlsx_filename: The Excel file 'xlsx' which you want to retrieve its data
    sheet_name: The name of the sheet in your Excel file
    last_row: The last row in your Excel file
    rows_per_patient: The number of rows used for each patient (sample)
    pilot_study: 1 if it is Pilot Study
'''
def create_df_clinical(xlsx_filename, sheet_name, last_row, rows_per_patient, pilot_study):
    folder_name = "ExcelFiles/"
    '''
        columns_headers ->  'MedMotNerLat', 'MedMotNerAmp', 'MedMotNerVel', ... , 'SenLowLimb-HP', 'VibUpper', 'VibLower'
        Stores the header names of columns [Clinical data]
        columns: the columns values
        patient_ids: all patient ids in this Excel file store in this list 
    '''
    columns_headers = []
    allcolumns = {}
    patient_ids = []
    try:
        columns_headers = pickle.load(open("NecessaryPickles/clinical_data_col_headers.pickle","rb"))
    except Exception as ex:
        print("'clinical_data_col_headers.pickle' file does not exist in NecessaryPickles directory!")
        return
    for i in range(len(columns_headers)):
        allcolumns[columns_headers[i]] = []
    '''Retrieve the Excel file'''
    sheet = pe.get_sheet(file_name=folder_name + xlsx_filename, sheet_name=sheet_name)
    for r in range(rows_per_patient, last_row + 1, rows_per_patient):
        '''
            patient_id: Patient (sample) name with its id
                (Be sure that your patient names are located in 'C' column of your Excel File)
            Right_12month: The row for clinical data after 12 months for Right side of the patient
            Left_12month: The row for clinical data after 12 months for Left side of the patient
        '''
        patient_id = sheet["C" + str(r - (rows_per_patient-3))].strip()
        if(patient_id == ''):
            break
        patient_ids.append(patient_id)
        Right_12month = sheet.row[r][11-pilot_study:46-pilot_study]
        Left_12month = sheet.row[r + 1][11-pilot_study:46-pilot_study]
        ''' Check the clinical data values '''
        col = 0
        for i in range(len(Right_12month)):
            r_num = l_num = ''
            final_num = 0
            ''' Because of lack of information for H-wave test, we just ignore it [position 26] '''
            if (i != 26):
                try:
                    ''' Try to convert the value to float '''
                    l_num = float(str(Left_12month[i]).strip())
                except Exception as ex:
                    ''' if there exists a '>' just remove it and remain the number '''
                    if ('>' in str(Left_12month[i])):
                        l_num = str(Left_12month[i]).strip()
                        l_num = int(l_num[1:])
                    else:
                        l_num = ''
                try:
                    r_num = float(str(Right_12month[i]).strip())
                except Exception as ex:
                    if ('>' in str(Right_12month[i])):
                        r_num = str(Right_12month[i]).strip()
                        r_num = int(r_num[1:])
                    else:
                        r_num = ''
                if (l_num == '' and r_num == ''):
                    print("ERROR occured... [The numbers for both left and right are incorrect] at " + str(i))
                    print("The patient is: " + patient_id)
                    exit(0)

                ''' If there is no value for Left side, we just put Right side value (vice versa) '''
                if (l_num == ''):
                    final_num = r_num
                elif (r_num == ''):
                    final_num = l_num
                else:
                    ''' If both Left and Right sides are available, we put the average value '''
                    final_num = (r_num + l_num) / 2
                allcolumns[columns_headers[col]].append(round(final_num,3))
                col += 1

    ''' Create pandas DataFrame of the Excel file [rows are patient ids and columns are examinations]'''
    clinical_df = pd.DataFrame(allcolumns, columns=columns_headers, index=patient_ids)

    latest_clinical_df = ""
    try:
        latest_clinical_df = pickle.load(open("ObtainedPickles/clinical_df.pickle", "rb"))
    except Exception as ex:
        pickle.dump(clinical_df, open("ObtainedPickles/clinical_df.pickle","wb"))
        return 0
    ''' Remove any duplicate index find in this data frame and just keep the first one '''
    clinical_df = latest_clinical_df.append(clinical_df).groupby(level=0).first()
    pickle.dump(clinical_df,open("ObtainedPickles/clinical_df.pickle", "wb"))
    return 0


'''
    Creates Data Frame of Output from 'ExcelFiles' folder
    The DataFrame includes binary values (Physician report for each patient)
    Each patient may belongs to multiple classes (1 or more classes)
    xlsx_filename: The Excel file 'xlsx' which you want to retrieve its data
    sheet_name: The name of the sheet in your Excel file
    last_row: The last row in your Excel file
    rows_per_patient: The number of rows used for each patient (sample)
    pilot_study: to see whether it is Pilot study or not
    num_of_neurologist: the maximum number of physicians (neurologists) who made the report
'''
def create_df_output(xlsx_filename,sheet_name,last_row, rows_per_patient, pilot_study, num_of_neurologist):
    '''
        classes include: PNS (Sensory Polyneuropathy), PNM (Motor Polyneuropathy), En (Entrapment Syndrome), Rd (Radiculopathy)
    '''
    classes = ["PNS","PNM","En","Rd"]
    folder_name = "ExcelFiles/"
    allcolumns = {}
    patient_ids = []
    for i in range(len(classes)):
        allcolumns[classes[i]] = []
    '''Retrieve the Excel file'''
    sheet = pe.get_sheet(file_name=folder_name + xlsx_filename, sheet_name=sheet_name)
    ''' The column number that physician reports start '''
    pindex_origninal = 77
    if(pilot_study == 1):
        pindex_origninal = 73
    for r in range(rows_per_patient, last_row + 1, rows_per_patient):
        patient_id = sheet["C" + str(r - (rows_per_patient-3))].strip()
        if(patient_id == ''):
            break
        '''
            Voting system! It searches for all physicians report and then make an average and decide which classes are include
        '''
        PNS = 0
        PNM = 0
        En = 0
        Rd = 0
        pindex = pindex_origninal
        ''' Number of phsicians who made report for current patient '''
        number_of_physician = 0
        ''' The final report (average of physician reports) '''
        physicians_final_opinion = []
        max_num_of_ph = num_of_neurologist
        while (max_num_of_ph > 0):
            '''Each Physician Opinion'''
            physician_opinion = sheet.row[r][pindex:pindex + 4]
            ''' Is there exist any report or not? '''
            is_allowed =  False
            for an_opinion in physician_opinion:
                if(an_opinion.lower() == "normal" or an_opinion.lower() == '+'):
                    is_allowed = True
            max_num_of_ph -= 1
            ''' To read the next neurologist report '''
            pindex += 4
            if(is_allowed):
                physician_opinion = [1 if x == '+' else 0 for x in physician_opinion]
                number_of_physician += 1
                PNS += physician_opinion[0]
                PNM += physician_opinion[1]
                En += physician_opinion[2]
                Rd += physician_opinion[3]
        ''' Average of opinions '''
        if (number_of_physician != 0):
            patient_ids.append(patient_id)
            PNS = PNS / number_of_physician
            physicians_final_opinion.append(PNS)
            PNM /= number_of_physician
            physicians_final_opinion.append(PNM)
            En /= number_of_physician
            physicians_final_opinion.append(En)
            Rd /= number_of_physician
            physicians_final_opinion.append(Rd)
            ''' Make it binary list '''
            physicians_final_opinion = [1 if x >= 0.5 else 0 for x in physicians_final_opinion]
            allcolumns["PNS"].append(physicians_final_opinion[0])
            allcolumns["PNM"].append(physicians_final_opinion[1])
            allcolumns["En"].append(physicians_final_opinion[2])
            allcolumns["Rd"].append(physicians_final_opinion[3])
    print(allcolumns)
    print(len(allcolumns["PNS"]))
    ''' Create pandas DataFrame of the Excel file [rows are patient ids and columns are classes]'''
    output_df = pd.DataFrame(allcolumns, columns=classes, index=patient_ids)
    latest_output_df = ""
    try:
        latest_output_df = pickle.load(open("ObtainedPickles/output_df.pickle", "rb"))
    except Exception as ex:
        pickle.dump(output_df, open("ObtainedPickles/output_df.pickle","wb"))
        return 0
    ''' Remove any duplicate index find in this data frame and just keep the first one '''
    output_df = latest_output_df.append(output_df).groupby(level=0).first()
    pickle.dump(output_df,open("ObtainedPickles/output_df.pickle", "wb"))
    return 0


'''
    Obtain the ACMG-AMP score for a specific mutation and returns the score
    iv: InterVar text
    clinvar_type: ClinVar text
'''
def calculate_ACMG_AMP_score(iv, clinvar_type):
    #ClinVar Score
    cv_score = 0
    if (clinvar_type.find("likely") != -1):
        if (clinvar_type.find("patho") != -1):
            #Likely pathogenic
            cv_score = 2.501
        else:
            #Likely benign
            cv_score = -4.999
    elif (clinvar_type.find("patho") != -1):
        #Pathogenic
        cv_score = 5.001
    elif (clinvar_type.find("beni") != -1):
        #Benign
        cv_score = -14.999
    else:
        #UNK
        cv_score = 0

    ''' Values to determine the severity of mutations '''
    ''' ------ Pathogenic values (POSITIVE VALUES) ------ '''
    PVS1 = 4.165  # Very Strong
    PS = 2.6  # Strong
    PM = 0.835  # Moderate
    PP = 0.45  # Supporting

    ''' ------ Benign values (NEGATIVE VALUES) ------ '''
    BA1 = - 5.5  # Stand-alone
    BS = - 3.7  # Strong
    BP = - 0.7  # Supporting
    interVar = iv.split(" PVS1=")
    type = interVar[0][11:]
    ''' Obtain the score for each mutation '''
    score = 0
    if (type.lower() == "likely pathogenic" or type.lower() == "pathogenic"):
        # PVS1 value
        score = score + (int(interVar[1][0]) * PVS1)

        # PS values
        rest = interVar[1]
        val = rest.split('[', 1)[1].split(']')[0]
        values = val.split(', ')
        for value in values:
            score = score + (int(value) * PS)

        # PM values
        rest = rest.split("PM=")[1]
        val = rest.split('[', 1)[1].split(']')[0]
        values = val.split(', ')
        for value in values:
            score = score + (int(value) * PM)

        # PP values
        rest = rest.split("PP=")[1]
        val = rest.split('[', 1)[1].split(']')[0]
        values = val.split(', ')
        for value in values:
            score = score + (int(value) * PP)

    elif (type.lower() == "likely benign" or type.lower() == "benign"):
        rest = interVar[1].split("BA1=")[1]
        score = score + (int(rest[0]) * BA1)

        # BS values
        rest = rest.split("BS=")[1]
        val = rest.split('[', 1)[1].split(']')[0]
        values = val.split(', ')
        for value in values:
            score = score + (int(value) * BS)

        # BP values
        rest = rest.split("BP=")[1]
        val = rest.split('[', 1)[1].split(']')[0]
        values = val.split(', ')
        for value in values:
            score = score + (int(value) * BP)
    else:
        # Assign zero for Uncertain of Significance
        score = 0
    #Return the best value for ACMG-AMP
    if(score != 0):
        if(cv_score == 0):
            return round(score, 3)
        else:
            if(score > cv_score):
                return round(score, 3)
            return cv_score
    return cv_score


'''
    Rescales the SIFT and CADD scores into ACMG-AMP standard
    a < c < b (old interval) ; y < x < z (new interval)
    value in the new scale will be x = [(c - a)*(z - y)/(b - a)] + y 
'''
def rescale_sift_cadd_scores(sift, cadd):
    scores = {}
    try:
        SIFT_SCORE = float(sift)
        c = -1 * SIFT_SCORE
        # Pathogenic and Likely Pathogenic interval (SIFT)
        if (SIFT_SCORE <= 0.05):
            scores["SIFT"] = ((c + 0.05) * 250) + 2.5
        # Benign and Likely Benign interval (SIFT)
        else:
            scores["SIFT"] = ((c + 1) * 14 / 0.95) - 15
    except:
        # Not Available (SIFT)
        scores["SIFT"] = 0

    try:
        CADD_SCORE = float(cadd)
        c = CADD_SCORE
        # Pathogenic and Likely Pathogenic interval (CADD)
        if (CADD_SCORE >= 15):
            scores["CADD"] = ((c - 15) * 0.5) + 2.5
        # Benign and Likely Benign interval (CADD)
        else:
            scores["CADD"] = (c * 14 / 15) - 15
    except:
        # Not Available (CADD)
        scores["CADD"] = 0
    return scores


'''
    Calculates scores for all genes in all patients (samples)
    intervar_dir: The address of the directory which you put your InterVar files inside
'''
def calculate_scores(intervar_dir):
    for intervarfile in listdir(intervar_dir):
        if (".intervar" in intervarfile):
            scores = {}
            patient_id = intervarfile[:-28]
            print(patient_id)
            count_SIFT = 0
            count_CADD = 0
            count_Others = 0
            with open(intervar_dir + intervarfile) as ivfile:
                reader = csv.DictReader(ivfile, dialect='excel-tab')
                for row in reader:
                    iv = row[' InterVar: InterVar and Evidence ']
                    clinvar_type = row['clinvar: Clinvar ']
                    score = calculate_ACMG_AMP_score(iv, clinvar_type)
                    other_scores = rescale_sift_cadd_scores(row['SIFT_score'], row['CADD_phred'])

                    #We try to reduce the number of Uncertain Significance
                    best_score = 0
                    #ACMGAMP=0
                    if(score == 0):
                        if(other_scores["SIFT"] == 0):
                            # ACMGAMP=0; SIFT=0; Whatever CADD
                            best_score = other_scores["CADD"]
                        else:
                            if(other_scores["CADD"] == 0):
                                #ACMGAMP=0; CADD=0; Whatever SIFT
                                best_score = other_scores["SIFT"]
                            else:
                                #ACMGAMP=0; max(CADD, SIFT)
                                best_score = other_scores["CADD"]
                                if(other_scores["SIFT"] > best_score):
                                    best_score = other_scores["SIFT"]
                    #ACMGAMP is available
                    else:
                        if(other_scores["SIFT"] == 0):
                            if(other_scores["CADD"] == 0):
                                #CADD=0; SIFT=0; Whatever ACMGAMP
                                best_score = score
                            else:
                                #SIFT=0; max(ACMGAMP, CADD)
                                best_score = score
                                if(other_scores["CADD"] > score):
                                    best_score = other_scores["CADD"]
                        else:
                            if(other_scores["CADD"] == 0):
                                #CADD=0; max(ACMGAMP, SIFT)
                                best_score = score
                                if(other_scores["SIFT"] > score):
                                    best_score = other_scores["SIFT"]
                            else:
                                #max(ACMGAMP, CADD, SIFT)
                                best_score = score
                                if(other_scores["SIFT"] > best_score):
                                    best_score = other_scores["SIFT"]
                                if(other_scores["CADD"] > best_score):
                                    best_score = other_scores["CADD"]

                    if (row['Ref.Gene'] in scores):
                        #Benign or Likely Benign till now
                        if (scores[row['Ref.Gene']] < 0):
                            #If we observe an UNK we ignore previous
                            if(best_score == 0):
                                scores[row['Ref.Gene']] = 0
                            #If we observe a Benign or Likely Benign, we choose the highest score
                            elif(best_score < 0):
                                if(scores[row['Ref.Gene']] < best_score):
                                    scores[row['Ref.Gene']] = best_score
                            #If we observe a P/LP then we ignore all previous UNK and B/LB and put this value
                            else:
                                scores[row['Ref.Gene']] = best_score
                        #UNK till now
                        elif(scores[row['Ref.Gene']] == 0):
                            #If there exists a Pathogenic or Likely Pathogenic, we accept
                            #Otherwise, we don't care!
                            if(score > 0):
                                scores[row['Ref.Gene']] = best_score
                        #Pathogenic or Likely Pathogenic till now
                        else:
                            #If we observe another P/LP mutation, then we add it to the previous values
                            if (best_score > 0):
                                scores[row['Ref.Gene']] = scores[row['Ref.Gene']] + best_score
                    else:
                        scores[row['Ref.Gene']] = best_score
            pickle.dump(scores, open("ScorePickles/" + patient_id + "-scores.pickle", "wb"))


'''
    Creates DataFrame of Score matrix obtained from InterVar analyzed data (stored in ScorePickles directory)
    (Before calling this function you should call 'calculate_scores' method)
'''
def create_df_scores():
    dir = "ScorePickles/"
    intersectedSet = []
    isFirstTime = True
    special_genes = []
    for file in listdir(dir):
        pmutations = pickle.load(open(dir+file,"rb"))
        length = len(set(pmutations.keys()))
        if(isFirstTime):
            isFirstTime = False
            intersectedSet = set(pmutations.keys())
        else:
            intersectedSet = intersectedSet.intersection(set(pmutations.keys()))
            a = set(pmutations.keys()).difference(intersectedSet)
            for aa in a:
                if(pmutations[aa] > 1):
                    special_genes.append(aa)
    intersectedSet = intersectedSet.union(set(special_genes))
    # Stores all Gene Names
    columns_headers = list(intersectedSet)
    # Store patients id for the DataFrame
    patient_ids = []
    # Stores all columns
    allcolumns = {}
    for col in columns_headers:
        allcolumns[col] = []
    for file in listdir(dir):
        patient_ids.append(file[:-14])
        pmutations = pickle.load(open(dir + file, "rb"))
        for gene in columns_headers:
            if(gene not in pmutations):
                allcolumns[gene].append(-15.001)
            else:
                allcolumns[gene].append(pmutations[gene])
    scores_df = pd.DataFrame(allcolumns, columns=columns_headers, index=patient_ids)

    latest_scores_df = ""
    try:
        latest_scores_df = pickle.load(open("ObtainedPickles/scores_df.pickle", "rb"))
    except Exception as ex:
        pickle.dump(scores_df, open("ObtainedPickles/scores_df.pickle","wb"))
        return 0
    # Remove any duplicate index find in this data frame and just keep the first one 
    scores_df = latest_scores_df.append(scores_df).groupby(level=0).first()
    pickle.dump(scores_df, open("ObtainedPickles/scores_df.pickle", "wb"))


'''
    patientsOf: can be either choose "eo" or "c" to retrieve the heatmap belongs to these patients
    genesFor: can be either choose "pl" or "ox" to retrieve the heatmaps belongs to these genes
    "eo" patients are the patients who received paclitaxel
    "c" patients are the patients who received oxaliplatin
'''
def chemotherapy_drugs_genes_heatmaps(patientsOf, genesFor):
    #These relevant genes found from different studies
    #Endometrial/Ovarian cancer patients received paclitaxel chemotherapy drugs (paclitaxel-related genes)
    eo_genes = ["ABCB1","TUBB2A","GSK3B","RRM1","FZD3","XKR4","RFX2","LYRM4","SCN10A","EPHA4","CCNH", "EPHA8", "CYP1B1","ICAM1","CASP9","IGF1R","FCRL2","CBS","MYO5A"]
    #Colon cancer patients received oxaliplatin chemotherapy drugs (oxaliplatin-related genes)
    c_genes = ["ERCC1","FOXC1","GSTP1","POU2AF1","ACYP2","SCN4A", "DPYD", "PELO", "FARS2", "VEGFA", "TDO2", "NFATC4","SERPINB2"]

    score_df = pickle.load(open("ObtainedPickles/complete_score_use_clinical_157patients_df.pickle", "rb"))
    first_df = score_df[c_genes]
    if(genesFor == "pl"):
        first_df = score_df[eo_genes]
    elif(genesFor == "ox"):
        first_df = score_df[c_genes]
    else:
        print("ERROR: The genesFor parameter must be 'pl' or 'ox'")
        exit(0)
    #Colon cancer patients
    c = []
    #Endometrial/Ovarian cancer patients
    eo = []
    for i in list(first_df.index):
        if i.find("C-") != -1:
            c.append(i)
        else:
            eo.append(i)
    p_df = ""
    if(patientsOf == "c"):
        p_df = first_df.loc[c]
    elif(patientsOf == "eo"):
        p_df = first_df.loc[eo]
    else:
        print("ERROR: The patientsOf parameter must be 'eo' or 'c'")
    sns.set(font_scale=0.5)
    ax = sns.clustermap(p_df, cmap='coolwarm',standard_scale=1, row_cluster=False, col_cluster=False, vmin=0.0, vmax=1.0)
    ax.savefig(patientsOf + "_patients_" + genesFor + "_genes.png", dpi=400)


'''
    Select the columns of 'first_df' which has high correlations with columns of 'second_df'
    first_df: The first DataFrame which we want to extract some columns
    second_df: The second DataFrame which is our criteria
    corr_method: You can choose one of the options (‘pearson’, ‘kendall’, ‘spearman’)
    N: The number of selected columns with high rank
    RETURNS a sub-DataFrame of first_df with N selected columns 
'''
def select_high_correlated_columns(first_df, second_df, corr_method, N):
    set_first = set(first_df.index.tolist())
    set_second = set(second_df.index.tolist())
    selected_rows = set_first.intersection(set_second)
    first_selected_df = first_df.loc[selected_rows]
    second_selected_df = second_df.loc[selected_rows]
    ''' Calculate the Correlation between two different DataFrames '''
    # Firstly, we need to concatenate these two DataFrames and then calculate the correlation
    corr = pd.concat([first_selected_df, second_selected_df], axis=1).corr(method=corr_method)
    # NOTICE: to calculate the score of each columns in the first DataFrame we convert all the correlation value to the absolute value
    scores_of_first_columns = abs(corr[second_selected_df.columns.tolist()].loc[first_selected_df.columns.tolist()]).sum(axis=1).to_frame()
    #print(scores_of_first_columns.sort_values(by=0, ascending=False))
    ''' Select top N features which highly correlated with columns of the second DataFrame '''
    topN_df = scores_of_first_columns.nlargest(N, 0)
    ''' Return a DataFrame (a sub-DataFrame of 'first_df') which has N highly correlated columns   '''
    return first_selected_df[topN_df.index.tolist()]


'''
    Plot the distribution of the number of Pathogenic/Likely Pathogenic genes in each patient
'''
def get_distribution_pathogenic_genes():
    scores_df = pickle.load(open("ObtainedPickles/scores_df.pickle", "rb"))
    distribution = scores_df.where(scores_df > 1).count(1)
    distribution = distribution.to_frame()
    distribution = distribution.rename({0: "Distribution"}, axis='columns')
    distribution = distribution.sort_values(by="Distribution")
    ax = distribution.plot.bar(rot=0)
    plt.show()


'''
    Predicts the gene significance Matrix unkown values and obtain the error and accuracy values by using Examination matrix
    This method uses two matrix factorizations (Patient-Gene matrix and Patient-Examination matrix) of Collaborative Filtering
'''
def predict_score_matrix_unkowns_CF_using_clinical():
    M_df = pickle.load(open("ObtainedPickles/scores_df.pickle","rb"))
    E_df = pickle.load(open("ObtainedPickles/clinical_df.pickle","rb"))
    set_m = set(M_df.index.tolist())
    set_e = set(E_df.index.tolist())
    selected_rows = set_m.intersection(set_e)
    M_selected_df = M_df.loc[selected_rows]
    E_selected_df = E_df.loc[selected_rows]

    genes = list(M_selected_df)
    gene_numbers = {}
    for j in range(len(genes)):
        gene_numbers[genes[j]] = j + 1
    patients = list(M_selected_df.index)
    patient_numbers = {}
    for j in range(len(patients)):
        patient_numbers[patients[j]] = j + 1
    M_selected_df = M_selected_df.rename(index=str, columns=gene_numbers)
    M_selected_df = M_selected_df.rename(patient_numbers, axis='index')

    df = M_selected_df.unstack().reset_index(name='value_pg')
    df = df.rename(columns={'level_0': 'gene', 'level_1': 'patient'})
    df = df.where(df['value_pg'] != 0).dropna()
    df = df.sample(frac=1)
    df = df.reset_index(drop=True).sort_index()
    #Calculate Z-normalization
    mean = df['value_pg'].mean()
    std = df['value_pg'].std()
    df['value_pg'] = (df['value_pg'] - df['value_pg'].mean())/df['value_pg'].std()

    examinations = list(E_selected_df)
    for examination in examinations:
        E_selected_df[examination] = (E_selected_df[examination] - E_selected_df[examination].mean())/E_selected_df[examination].std()
    examination_numbers = {}
    for j in range(len(examinations)):
        examination_numbers[examinations[j]] = j + 1
    E_selected_df = E_selected_df.rename(index=str, columns=examination_numbers)
    E_selected_df = E_selected_df.rename(patient_numbers, axis='index')
    df2 = E_selected_df.unstack().reset_index(name='value_pe')
    df2 = df2.rename(columns={'level_0': 'examination', 'level_1': 'patient'})
    array1to34 = []
    j = 1
    for i in range(len(df.axes[0])):
        if(j == 35):
            j = 1
        array1to34.append(j)
        j += 1
    df["examination"] = array1to34
    result_df = pd.merge(df, df2, how='inner', on=['patient', 'examination'])
    result_df = result_df.sample(frac=1)
    result_df = result_df.reset_index(drop=True).sort_index()
    info = {}
    for test_size in [0.2, 0.3]:
        info["test" + str(test_size)] = {}
        for n_latent_factors in range(10,201,10):
            info["test" + str(test_size)]["n" + str(n_latent_factors)] = {}
            info["test" + str(test_size)]["n" + str(n_latent_factors)]['error'] = []
            info["test" + str(test_size)]["n" + str(n_latent_factors)]['accuracy'] = []
            for rep in range(10):
                train, test = train_test_split(result_df, test_size=test_size)
                n_examinations = len(examinations)
                n_patients = len(patients)
                n_genes = len(genes)
                examination_input = keras.layers.Input(shape=[1], name='Examination')
                examination_embedding = keras.layers.Embedding(n_examinations + 1, n_latent_factors, name='Examination-Embedding')(examination_input)
                examination_vec = keras.layers.Flatten(name='FlattenExaminations')(examination_embedding)
                gene_input = keras.layers.Input(shape=[1],name='Gene')
                gene_embedding = keras.layers.Embedding(n_genes + 1, n_latent_factors, name='Gene-Embedding')(gene_input)
                gene_vec = keras.layers.Flatten(name='FlattenGenes')(gene_embedding)
                patient_input = keras.layers.Input(shape=[1],name='Patient')
                patient_embedding = keras.layers.Embedding(n_patients + 1, n_latent_factors,name='Patient-Embedding')(patient_input)
                patient_vec = keras.layers.Flatten(name='FlattenPatients')(patient_embedding)
                prod = keras.layers.Dot(axes=-1)([gene_vec, patient_vec])
                prod2 = keras.layers.Dot(axes=-1)([examination_vec, patient_vec])
                model = keras.Model([patient_input, gene_input, examination_input], [prod, prod2])
                model.compile('adam', 'mean_squared_error')
                model.fit([train.patient, train.gene, train.examination], [train.value_pg, train.value_pe], batch_size=64, epochs=10, verbose=1)
                y_hat = model.predict([test.patient, test.gene, test.examination])
                y_true = test.value_pg
                error = mean_absolute_error(y_true, y_hat[0])
                info["test" + str(test_size)]["n" + str(n_latent_factors)]['error'].append(error)
                y = list(y_true)
                count_correct = 0
                for i in range(len(y)):
                    a = y[i]*std + mean
                    b = y_hat[0][i]*std + mean
                    k = a*b
                    if(k > 0):
                        count_correct += 1
                correct_percentage = (count_correct/len(y))*100
                info["test" + str(test_size)]["n" + str(n_latent_factors)]['accuracy'].append(correct_percentage)
                print("Test Size: " + str(test_size) + " , num of latent: " + str(n_latent_factors) + " Error: " + str(error) + " , Accuracy: " + str(correct_percentage))
    pickle.dump(info, open("percentages-two-matrices.pickle","wb"))


'''
    Predicts the gene significance Matrix unkown values and obtain the error and accuracy values 
    This method uses Matrix Factorization based Collaborative Filtering
'''
def predict_score_matrix_unkowns_CF():
    M_df = pickle.load(open("ObtainedPickles/scores_df.pickle", "rb"))
    genes = list(M_df)
    gene_numbers = {}
    for j in range(len(genes)):
        gene_numbers[genes[j]] = j + 1
    patients = list(M_df.index)
    patient_numbers = {}
    for j in range(len(patients)):
        patient_numbers[patients[j]] = j + 1
    M_df = M_df.rename(index=str, columns=gene_numbers)
    M_df = M_df.rename(patient_numbers, axis='index')

    df = M_df.unstack().reset_index(name='value')
    df = df.rename(columns={'level_0': 'gene', 'level_1': 'patient'})
    df = df.where(df['value'] != 0).dropna()
    df = df.sample(frac=1)
    df = df.reset_index(drop=True).sort_index()
    mean = df['value'].mean()
    std = df['value'].std()
    df['value'] = (df['value'] - df['value'].mean()) / df['value'].std()
    info = {}
    for test_size in [0.2, 0.3]:
        info["test" + str(test_size)] = {}
        for n_latent_factors in range(10, 201, 10):
            info["test" + str(test_size)]["n" + str(n_latent_factors)] = {}
            info["test" + str(test_size)]["n" + str(n_latent_factors)]['error'] = []
            info["test" + str(test_size)]["n" + str(n_latent_factors)]['accuracy'] = []
            for rep in range(10):
                train, test = train_test_split(df, test_size=test_size)
                n_patients = len(patients)
                n_genes = len(genes)

                gene_input = keras.layers.Input(shape=[1], name='Gene')
                gene_embedding = keras.layers.Embedding(n_genes + 1, n_latent_factors, name='Gene-Embedding')(
                    gene_input)
                gene_vec = keras.layers.Flatten(name='FlattenGenes')(gene_embedding)

                patient_input = keras.layers.Input(shape=[1], name='Patient')
                patient_embedding = keras.layers.Embedding(n_patients + 1, n_latent_factors, name='Patient-Embedding')(
                    patient_input)
                patient_vec = keras.layers.Flatten(name='FlattenPatients')(patient_embedding)

                prod = keras.layers.Dot(axes=-1)([gene_vec, patient_vec])
                model = keras.Model([patient_input, gene_input], prod)
                model.summary()
                model.compile('adam', 'mean_squared_error')
                model.fit([train.patient, train.gene], train.value, batch_size=128, epochs=10, verbose=1)
                y_hat = model.predict([test.patient, test.gene])
                y_true = test.value
                error = mean_absolute_error(y_true, y_hat)
                info["test" + str(test_size)]["n" + str(n_latent_factors)]['error'].append(error)
                y = list(y_true)
                count_correct = 0
                for i in range(len(y)):
                    a = y[i] * std + mean
                    b = y_hat[i] * std + mean
                    k = a * b
                    if (k > 0):
                        count_correct += 1
                correct_percentage = (count_correct / len(y)) * 100
                info["test" + str(test_size)]["n" + str(n_latent_factors)]['accuracy'].append(correct_percentage)
                print("Test Size: " + str(test_size) + " , num of latent: " + str(n_latent_factors) + " Error: " + str(
                    error) + " , Accuracy: " + str(correct_percentage))
    pickle.dump(info, open("percentages.pickle", "wb"))


'''
    Predict Score Matrix unkown values and obtain the error and accuracy values [You must determine test size and number of latent factors]
    This method uses Matrix Factorization based Collaborative Filtering
    test_size: 0.2 or 0.3 (fraction of dataset)
    n_latent_factors: The length of latent vectors
'''
def predict_unkowns_using_clinical(test_size, n_latent_factors):
    M_df = pickle.load(open("ObtainedPickles/scores_df.pickle", "rb"))
    E_df = pickle.load(open("ObtainedPickles/clinical_df.pickle", "rb"))
    set_m = set(M_df.index.tolist())
    set_e = set(E_df.index.tolist())
    selected_rows = set_m.intersection(set_e)
    M_selected_df = M_df.loc[selected_rows]
    E_selected_df = E_df.loc[selected_rows]
    genes = list(M_selected_df)
    gene_numbers = {}
    for j in range(len(genes)):
        gene_numbers[genes[j]] = j + 1
    patients = list(M_selected_df.index)
    patient_numbers = {}
    for j in range(len(patients)):
        patient_numbers[patients[j]] = j + 1
    M_selected_df = M_selected_df.rename(index=str, columns=gene_numbers)
    M_selected_df = M_selected_df.rename(patient_numbers, axis='index')
    df = M_selected_df.unstack().reset_index(name='value_pg')
    df = df.rename(columns={'level_0': 'gene', 'level_1': 'patient'})
    df = df.where(df['value_pg'] != 0).dropna()
    df = df.sample(frac=1)
    df = df.reset_index(drop=True).sort_index()
    mean = df['value_pg'].mean()
    std = df['value_pg'].std()
    df['value_pg'] = (df['value_pg'] - df['value_pg'].mean()) / df['value_pg'].std()
    examinations = list(E_selected_df)
    for examination in examinations:
        E_selected_df[examination] = (E_selected_df[examination] - E_selected_df[examination].mean()) / E_selected_df[
            examination].std()
    examination_numbers = {}
    for j in range(len(examinations)):
        examination_numbers[examinations[j]] = j + 1
    E_selected_df = E_selected_df.rename(index=str, columns=examination_numbers)
    E_selected_df = E_selected_df.rename(patient_numbers, axis='index')
    df2 = E_selected_df.unstack().reset_index(name='value_pe')
    df2 = df2.rename(columns={'level_0': 'examination', 'level_1': 'patient'})
    array1to34 = []
    j = 1
    for i in range(len(df.axes[0])):
        if (j == 35):
            j = 1
        array1to34.append(j)
        j += 1
    df["examination"] = array1to34
    result_df = pd.merge(df, df2, how='inner', on=['patient', 'examination'])
    result_df = result_df.sample(frac=1)
    result_df = result_df.reset_index(drop=True).sort_index()
    train, test = train_test_split(result_df, test_size=test_size)
    n_examinations = len(examinations)
    n_patients = len(patients)
    n_genes = len(genes)
    examination_input = keras.layers.Input(shape=[1], name='Examination')
    examination_embedding = keras.layers.Embedding(n_examinations + 1, n_latent_factors,
                                                   name='Examination-Embedding')(examination_input)
    examination_vec = keras.layers.Flatten(name='FlattenExaminations')(examination_embedding)
    gene_input = keras.layers.Input(shape=[1], name='Gene')
    gene_embedding = keras.layers.Embedding(n_genes + 1, n_latent_factors, name='Gene-Embedding')(
        gene_input)
    gene_vec = keras.layers.Flatten(name='FlattenGenes')(gene_embedding)
    patient_input = keras.layers.Input(shape=[1], name='Patient')
    patient_embedding = keras.layers.Embedding(n_patients + 1, n_latent_factors, name='Patient-Embedding')(
        patient_input)
    patient_vec = keras.layers.Flatten(name='FlattenPatients')(patient_embedding)
    prod = keras.layers.Dot(axes=-1)([gene_vec, patient_vec])
    prod2 = keras.layers.Dot(axes=-1)([examination_vec, patient_vec])
    model = keras.Model([patient_input, gene_input, examination_input], [prod, prod2])
    model.summary()
    model.compile('adam', 'mean_squared_error')
    model.fit([train.patient, train.gene, train.examination], [train.value_pg, train.value_pe],
              batch_size=64, epochs=10, verbose=1)
    y_hat = model.predict([test.patient, test.gene, test.examination])
    y_true = test.value_pg
    error = mean_absolute_error(y_true, y_hat[0])
    y = list(y_true)
    count_correct = 0
    for i in range(len(y)):
        a = y[i] * std + mean
        b = y_hat[0][i] * std + mean
        k = a * b
        if (k > 0):
            count_correct += 1
    correct_percentage = (count_correct / len(y)) * 100
    print(" Error: " + str(error) + " , Accuracy: " + str(correct_percentage))
    return model


'''
    Fills the gene significance matrix according the model that you predicted unknown values
    model: The model that you used to predict unknown values
'''
def fill_unk_score_matrix(model):
    M_df = pickle.load(open("ObtainedPickles/scores_df.pickle", "rb"))
    E_df = pickle.load(open("ObtainedPickles/clinical_df.pickle", "rb"))
    set_m = set(M_df.index.tolist())
    set_e = set(E_df.index.tolist())
    selected_rows = set_m.intersection(set_e)
    M_selected_df = M_df.loc[selected_rows]
    E_selected_df = E_df.loc[selected_rows]
    genes = list(M_selected_df)
    gene_numbers = {}
    for j in range(len(genes)):
        gene_numbers[genes[j]] = j + 1
    patients = list(M_selected_df.index)
    patient_numbers = {}
    for j in range(len(patients)):
        patient_numbers[patients[j]] = j + 1
    M_selected_df = M_selected_df.rename(index=str, columns=gene_numbers)
    M_selected_df = M_selected_df.rename(patient_numbers, axis='index')
    #Non-unk Members of the Score matrix
    df = M_selected_df.unstack().reset_index(name='value_pg')
    df = df.rename(columns={'level_0': 'gene', 'level_1': 'patient'})
    df = df.where(df['value_pg'] != 0).dropna()
    df = df.sample(frac=1)
    df = df.reset_index(drop=True).sort_index()
    mean = df['value_pg'].mean()
    std = df['value_pg'].std()
    df['value_pg'] = (df['value_pg'] - df['value_pg'].mean()) / df['value_pg'].std()
    #Members of the Clinical matrix
    examinations = list(E_selected_df)
    for examination in examinations:
        E_selected_df[examination] = (E_selected_df[examination] - E_selected_df[examination].mean()) / E_selected_df[
            examination].std()
    examination_numbers = {}
    for j in range(len(examinations)):
        examination_numbers[examinations[j]] = j + 1
    E_selected_df = E_selected_df.rename(index=str, columns=examination_numbers)
    E_selected_df = E_selected_df.rename(patient_numbers, axis='index')
    df2 = E_selected_df.unstack().reset_index(name='value_pe')
    df2 = df2.rename(columns={'level_0': 'examination', 'level_1': 'patient'})
    array1to34 = []
    j = 1
    for i in range(len(df.axes[0])):
        if (j == 35):
            j = 1
        array1to34.append(j)
        j += 1
    df["examination"] = array1to34
    known_df = pd.merge(df, df2, how='inner', on=['patient', 'examination'])
    known_df = known_df.sample(frac=1)
    known_df = known_df.reset_index(drop=True).sort_index()
    #Unkown members of the Score matrix
    unk_df = M_selected_df.unstack().reset_index(name='value_pg')
    unk_df = unk_df.rename(columns={'level_0': 'gene', 'level_1': 'patient'})
    unk_df = unk_df.where(unk_df['value_pg'] == 0).dropna()
    unk_df = unk_df.sample(frac=1)
    unk_df = unk_df.reset_index(drop=True).sort_index()
    array1to34 = []
    j = 1
    for i in range(len(unk_df.axes[0])):
        if (j == 35):
            j = 1
        array1to34.append(j)
        j += 1
    unk_df["examination"] = array1to34
    unknown_df = pd.merge(unk_df, df2, how='inner', on=['patient', 'examination'])
    unknown_df = unknown_df.sample(frac=1)
    unknown_df = unknown_df.reset_index(drop=True).sort_index()
    y_hat = model.predict([unknown_df.patient, unknown_df.gene, unknown_df.examination])
    unknown_df.value_pg = y_hat[0]
    #The score matrix with all values! (unkown values replaced by predicted values)
    complete_score_M_df = pd.concat([unknown_df, known_df]).reset_index().drop(['index'], axis=1)
    complete_score_M_df = complete_score_M_df.drop(['examination', 'value_pe'], axis=1)
    complete_score_M_df = complete_score_M_df.rename(index=str, columns={"value_pg": "value"})
    pnumbers = list(patient_numbers.values())
    pnumbers.sort()
    gnumbers = list(gene_numbers.values())
    gnumbers.sort()
    data = np.zeros((len(pnumbers), len(gnumbers)))
    for row in zip(complete_score_M_df['patient'], complete_score_M_df['gene'], complete_score_M_df['value']):
        rowindex = pnumbers.index(row[0])
        colindex = gnumbers.index(row[1])
        data[rowindex][colindex] = row[2]
    # The new complete score matrix
    newf = pd.DataFrame(data)
    newf.columns = gnumbers
    newf.index = pnumbers
    patient_ids_dict = dict(zip(patient_numbers.values(), patient_numbers.keys()))
    gene_names_dict = dict(zip(gene_numbers.values(), gene_numbers.keys()))
    #Replace the names of columns and rows
    newf = newf.rename(index=patient_ids_dict, columns=gene_names_dict)
    return newf


'''
    Feedforward neural network model (containing 3 hidden layers)
    input_size: The number of selected genes
'''
def nn_model(input_size):
    model = Sequential()
    model.add(Dense(512,input_dim=input_size,activation='tanh'))
    model.add(Dense(256,activation='tanh'))
    model.add(Dense(64,activation='tanh'))
    model.add(Dense(4,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


'''
    Calculates correlations (output labels - Clinical matrix - Gene Significance matrix)
    Select topN genes (In this case 1297 genes) according to the correlations
    Test the result dataset on 5 different types of multi label classification
    KFold_k: Number of folds in KFold evaluation scenario (default value = 5)
    colasp: 1 (use CoLaSp as the predictor) and 0 (use SMFM as the predictor) (default value = 1)
'''
def multi_label_classifications(KFold_k = 5, colasp = 1):
    #Retrieve the dataframes (Score matrix, Clinical matrix, and Output labels)
    score_df = pickle.load(open("ObtainedPickles/complete_score_use_clinical_df.pickle","rb"))
    if(colasp == 0):
        score_df = pickle.load(open("ObtainedPickles/complete_score_use_clinical_df.pickle", "rb"))
    output_df = pickle.load(open("ObtainedPickles/output2_df","rb"))
    clinical_df = pickle.load(open("ObtainedPickles/clinical2_df","rb"))
    cipn_genes = pickle.load(open("NecessaryPickles/cipn_genes.pickle", "rb"))
    #Find the correlation between Clinical matrix and Output labels
    set1 = set(clinical_df.index.tolist())
    set2 = set(output_df.index.tolist())
    selected_common_rows = set1.intersection(set2)
    first_df = clinical_df.loc[selected_common_rows]
    second_df = output_df.loc[selected_common_rows]
    corr = pd.concat([first_df, second_df], axis=1).corr(method="pearson")
    corrs = abs(corr[second_df.columns.tolist()].loc[first_df.columns.tolist()])
    for N_Clinical in range(2,10):
        #Select Top N_Clinical examinations which highly-correlated with PNS, PNM, En, and Rd
        pns_selected_clinical = list(corrs[["PNS"]].sum(axis=1).to_frame().nlargest(N_Clinical, 0).index)
        pnm_selected_clinical = list(corrs[["PNM"]].sum(axis=1).to_frame().nlargest(N_Clinical, 0).index)
        en_selected_clinical = list(corrs[["En"]].sum(axis=1).to_frame().nlargest(N_Clinical, 0).index)
        rd_selected_clinical = list(corrs[["Rd"]].sum(axis=1).to_frame().nlargest(N_Clinical, 0).index)
        selected_clinical_by_output = [pns_selected_clinical, pnm_selected_clinical, en_selected_clinical,
                                       rd_selected_clinical]
        merged = list(itertools.chain.from_iterable(selected_clinical_by_output))
        set_merged = set(merged)
        clinical_output_df = first_df[list(set_merged)]
        #Find the correlation between Clinical-output selected dataframe and Score dataframe
        set_first = set(score_df.index.tolist())
        set_second = set(clinical_output_df.index.tolist())
        selected_rows = set_first.intersection(set_second)
        first_selected_df = score_df.loc[selected_rows]
        second_selected_df = clinical_output_df.loc[selected_rows]
        corr = pd.concat([first_selected_df, second_selected_df], axis=1).corr(method="pearson")
        corrs = abs(corr[second_selected_df.columns.tolist()].loc[first_selected_df.columns.tolist()])
        for N_genes in range(80,220,10):
            #Extract the selected gene names
            imp_genes = []
            for examination in set_merged:
                imp_genes.append(list(corrs[[examination]].sum(axis=1).to_frame().nlargest(N_genes, 0).index))
            imp_genes.append(cipn_genes)
            imp_genes = list(itertools.chain.from_iterable(imp_genes))
            imp_genes = set(imp_genes)
            #Define KFold for evaluation
            kf = KFold(n_splits=KFold_k)
            final_score_df = first_selected_df[list(imp_genes)]
            final_score_df.sort_index(inplace=True)
            output_df.sort_index(inplace=True)
            #Obtain X and Y for evaluation
            X = final_score_df.values
            Y = output_df.values
            #Test 5 different classifiers on our dataset
            avg_f1 = 0
            avg_hamm_loss = 0
            avg_b_f1 = 0
            avg_b_hamm_loss = 0
            avg_ps_f1 = 0
            avg_ps_hamm_loss = 0
            avg_ffnn_f1 = 0
            avg_ffnn_hamm_loss = 0
            avg_svc_f1 = 0
            avg_svc_hamm = 0
            avg_ffnn_roc_auc = 0
            avg_roc_auc = 0
            avg_b_roc_auc = 0
            avg_ps_roc_auc = 0
            avg_svc_roc_auc = 0
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                # MLkNN
                parameters = {'k': range(1, 5), 's': [0.5, 0.7, 1.0]}
                score = 'f1_micro'
                classifier = GridSearchCV(MLkNN(), parameters, scoring=score)
                classifier.fit(X_train, Y_train)
                y_hat = classifier.predict(X_test)
                hamm = metrics.hamming_loss(Y_test, y_hat)
                f1 = metrics.f1_score(Y_test, y_hat, average='micro')
                avg_f1 += f1
                avg_hamm_loss += hamm
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                y_hat = y_hat.todense()
                a = y_hat.ravel()
                a = np.reshape(np.ravel(a), (len(Y_test) * 4,))
                fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), a)
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                avg_roc_auc += roc_auc["micro"]
                # BRkNNa
                parameters = {'k': range(3, 5)}
                score = 'f1_micro'
                classifier = GridSearchCV(BRkNNaClassifier(), parameters, scoring=score)
                classifier.fit(X_train, Y_train)
                y_hat = classifier.predict(X_test)
                cc_f1 = metrics.f1_score(Y_test, y_hat, average='micro')
                cc_hamm = metrics.hamming_loss(Y_test, y_hat)
                avg_b_f1 += cc_f1
                avg_b_hamm_loss += cc_hamm
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                y_hat = y_hat.todense()
                a = y_hat.ravel()
                a = np.reshape(np.ravel(a), (len(Y_test) * 4,))
                fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), a)
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                avg_b_roc_auc += roc_auc["micro"]
                # POWERSET - RandomForest
                classifier = LabelPowerset(
                    classifier=RandomForestClassifier(),
                    require_dense=[False, True]
                )
                classifier.fit(X_train, Y_train)
                y_hat = classifier.predict(X_test)
                lp_f1 = metrics.f1_score(Y_test, y_hat, average='micro')
                lp_hamm = metrics.hamming_loss(Y_test, y_hat)
                avg_ps_f1 += lp_f1
                avg_ps_hamm_loss += lp_hamm
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                y_hat = y_hat.todense()
                a = y_hat.ravel()
                a = np.reshape(np.ravel(a), (len(Y_test) * 4,))
                fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), a)
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                avg_ps_roc_auc += roc_auc["micro"]
                # SVC (Support Vector Classifier)
                clf = BinaryRelevance(
                    classifier=SVC(),
                    require_dense=[False, True]
                )
                clf.fit(X_train, Y_train)
                y_hat = clf.predict(X_test)
                hm = metrics.hamming_loss(Y_test, y_hat)
                fone = metrics.f1_score(Y_test, y_hat, average='micro')
                avg_svc_f1 += fone
                avg_svc_hamm += hm
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                y_hat = y_hat.todense()
                a = y_hat.ravel()
                a = np.reshape(np.ravel(a), (len(Y_test) * 4,))
                fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), a)
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                avg_svc_roc_auc += roc_auc["micro"]
                # Feedforward Neural Network (With 3 hidden layers)
                model = nn_model(len(imp_genes))
                y_hat = model.fit(X_train, Y_train, epochs=10, shuffle=True, validation_data=[X_test, Y_test], batch_size=10)
                scores = model.evaluate(X_test, Y_test)
                print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
                Y_pred = model.predict(X_test)
                ffnn_f1 = metrics.f1_score(Y_test, np.around(Y_pred), average='micro')
                ffnn_hamm = metrics.hamming_loss(Y_test, np.around(Y_pred))
                avg_ffnn_f1 += ffnn_f1
                avg_ffnn_hamm_loss += ffnn_hamm
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(4):
                    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_pred[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_pred.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                avg_ffnn_roc_auc += roc_auc["micro"]
            avg_ffnn_f1 /= KFold_k
            avg_ffnn_hamm_loss /= KFold_k
            avg_ffnn_roc_auc /= KFold_k
            avg_roc_auc /= KFold_k
            avg_b_roc_auc /= KFold_k
            avg_ps_roc_auc /= KFold_k
            avg_svc_roc_auc /= KFold_k
            avg_f1 /= KFold_k
            avg_hamm_loss /= KFold_k
            avg_b_f1 /= KFold_k
            avg_b_hamm_loss /= KFold_k
            avg_ps_f1 /= KFold_k
            avg_ps_hamm_loss /= KFold_k
            avg_svc_f1 /= KFold_k
            avg_svc_hamm /= KFold_k
            print("==============MLkNN=================================")
            print('Hamming Loss= ', str(avg_hamm_loss), ', F1 score= ', str(avg_f1), ', AUC= ', str(avg_roc_auc))
            print("==============BRkNNa================================")
            print('Hamming Loss= ', str(avg_b_hamm_loss), ', F1 score= ', str(avg_b_f1), ', AUC= ', str(avg_b_roc_auc))
            print("==============Random Forest=========================")
            print('Hamming Loss= ', str(avg_ps_hamm_loss), ', F1 score= ', str(avg_ps_f1), ', AUC= ', str(avg_ps_roc_auc))
            print("==============SVC===================================")
            print('Hamming Loss= ', str(avg_svc_hamm), ', F1 score= ', str(avg_svc_f1), ', AUC= ', str(avg_svc_roc_auc))
            print("==============FeedForward Neural Network============")
            print('Hamming Loss= ', str(avg_ffnn_hamm_loss), ', F1 score= ', str(avg_ffnn_f1), ', AUC= ', str(avg_ffnn_roc_auc))


'''
    It finds the maximum correlation of each CIPN gene with a selected examination in Clinical matrix 
    topN_correlated_genes: Show top N correlated genes in Line graph (default value = 22 "correlation bigger than 0.6")
    correlation bigger than 0.5 [topN_correlated_genes = 40]
    N_Clinical: Number of examinations that you want select which are high-correlated with the output labels (default value = 5)
    colasp: 1 (use CoLaSp as the predictor) and 0 (use SMFM as the predictor)
'''
def cipn_genes_correlations(N_Clinical = 5, topN_correlated_genes = 22, colasp=1):
    # Retrieve the dataframes (Score matrix, Clinical matrix, and Output labels)
    score_df = pickle.load(open("ObtainedPickles/complete_score_use_clinical_df.pickle", "rb"))
    if(colasp == 0):
        score_df = pickle.load(open("ObtainedPickles/complete_score_matrix.pickle", "rb"))
    output_df = pickle.load(open("ObtainedPickles/output_df.pickle", "rb"))
    clinical_df = pickle.load(open("ObtainedPickles/clinical_df.pickle", "rb"))
    cipn_genes = pickle.load(open("NecessaryPickles/cipn_genes.pickle", "rb"))
    set1 = set(clinical_df.index.tolist())
    set2 = set(output_df.index.tolist())
    selected_common_rows = set1.intersection(set2)
    first_df = clinical_df.loc[selected_common_rows]
    second_df = output_df.loc[selected_common_rows]
    corr = pd.concat([first_df, second_df], axis=1).corr(method="pearson")
    corrs = abs(corr[second_df.columns.tolist()].loc[first_df.columns.tolist()])
    pns_selected_clinical = list(corrs[["PNS"]].sum(axis=1).to_frame().nlargest(N_Clinical, 0).index)
    pnm_selected_clinical = list(corrs[["PNM"]].sum(axis=1).to_frame().nlargest(N_Clinical, 0).index)
    en_selected_clinical = list(corrs[["En"]].sum(axis=1).to_frame().nlargest(N_Clinical, 0).index)
    rd_selected_clinical = list(corrs[["Rd"]].sum(axis=1).to_frame().nlargest(N_Clinical, 0).index)
    selected_clinical_by_output = [pns_selected_clinical, pnm_selected_clinical, en_selected_clinical,
                                   rd_selected_clinical]
    merged = list(itertools.chain.from_iterable(selected_clinical_by_output))
    set_merged = set(merged)
    clinical_output_df = first_df[list(set_merged)]
    set_first = set(score_df.index.tolist())
    set_second = set(clinical_output_df.index.tolist())
    selected_rows = set_first.intersection(set_second)
    first_selected_df = score_df.loc[selected_rows]
    second_selected_df = clinical_output_df.loc[selected_rows]
    corr = pd.concat([first_selected_df, second_selected_df], axis=1).corr(method="pearson")
    corrs = abs(corr[second_selected_df.columns.tolist()].loc[first_selected_df.columns.tolist()])
    a = corrs.filter(regex='^LIN', axis=0)
    i1 = a.index
    corrs = corrs.drop(i1)
    a = corrs.filter(regex='^LOC', axis=0)
    i1 = a.index
    corrs = corrs.drop(i1)
    a = corrs.filter(regex='^MIR', axis=0)
    i1 = a.index
    corrs = corrs.drop(i1)
    a = corrs.filter(regex='^ZNF', axis=0)
    i1 = a.index
    corrs = corrs.drop(i1)
    cipn_corrs = corrs.loc[cipn_genes].max(axis=1).to_frame().sort_values(by=0, ascending=False)
    #Box Plot of CIPN genes correlation
    fig1, ax1 = plt.subplots()
    ax1.set_title('CIPN genes')
    ax1.boxplot(cipn_corrs[0])
    ax1.set_ylim(0, 1)
    if(colasp == 1):
        plt.savefig("boxplot-cipn-genes-CoLaSp.png", dpi=1000)
    else:
        plt.savefig("boxplot-cipn-genes-SMFM.png", dpi=1000)
    #Plot Line graph of topN_correlated_genes
    df = cipn_corrs.nlargest(topN_correlated_genes, 0)
    df = df.reset_index()
    fig, ax = plt.subplots()
    if (colasp == 1):
        ax.plot(df['index'], df[0], linestyle='-', marker='o', color='green', markersize=2)
    else:
        ax.plot(df['index'], df[0], linestyle='-', marker='o', color='red', markersize=2)
    ax.xaxis.set_tick_params(labelsize=4)
    ax.yaxis.set_tick_params(labelsize=6)
    # ax.set_ylim(0, 1)
    plt.xticks(rotation=90)
    if(colasp == 1):
        plt.savefig("topNcorrelated-cipn-genes-CoLaSp.png", dpi=1000)
    else:
        plt.savefig("topNcorrelated-cipn-genes-SMFM.png", dpi=1000)


'''
    It plots the accuracies of different latent dimensions
'''
def plot_latent_dimension_accuracy():
    A = pickle.load(open("ObtainedPickles/AccuracyErrorCF-CIPN.pickle","rb"))
    scores = []
    num = []
    for i in range(10, 201, 10):
        l = A['test0.2']['n' + str(i)]['accuracy']
        avg = 0
        for j in range(10):
            avg += l[j]
        avg /= 10
        scores.append(avg)
        num.append(i)
    x_pos = np.arange(len(num))

    bar = plt.bar(x_pos, scores, align='center', alpha=0.5)
    bar[0].set_color('r')
    plt.xticks(x_pos, num)
    plt.ylim(91, 95)
    plt.ylabel('Accuracy')
    plt.xlabel('Latent space dimension')
    plt.xticks(rotation=90)
    plt.savefig("latent-space-accuracies.png", dpi=1000)
