# CoLaSpSMFM
CoLaSp and SMFM are Matrix Factorization-based techniques to assign pathogenicity score to each gene and predict uncertain significance genes.
Inputs:
A folder name that contains the InterVar output files (.intervar)
Clinical dataset (Excel format) if you want use CoLaSp model (It is unnecessary if you want to use SMFM model)

Output:
A n by m matrix called the gene significance matrix. Where n is the number of samples (or patients) and m is the total number of genes. The matrix contains gene significance scores without any uncertain significance members.

Requirements (Latest Versions):
keras
pytorch
sklearn
pickle
pyexcel
skmultilearn
numpy
pandas
matplotlib
seaborn
