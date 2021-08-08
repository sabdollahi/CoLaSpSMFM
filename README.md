# CoLaSpSMFM
<b>CoLaSp</b> and <b>SMFM</b> are Matrix Factorization-based techniques to assign pathogenicity score to each gene and predict uncertain significance genes.
Inputs: <br>
A folder name that contains the InterVar output files (.intervar) <br>
Clinical dataset (Excel format) if you want use CoLaSp model (It is unnecessary if you want to use SMFM model) <br>

Output: <br>
A n by m matrix called the gene significance matrix. Where n is the number of samples (or patients) and m is the total number of genes. The matrix contains gene significance scores without any uncertain significance members. <br>

Requirements (Latest Versions): <br>
<ol>
  <li>keras</li>
  <li>pytorch</li>
  <li>sklearn</li>
  <li>pickle</li>
  <li>pyexcel</li>
  <li>skmultilearn</li>
  <li>numpy</li>
  <li>pandas</li>
  <li>matplotlib</li>
  <li>seaborn</li>
</ol> 
