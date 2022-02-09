######################
# Import libraries
######################
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
import math

######################
# Custom function
######################
## Calculate molecular descriptors
def AromaticProportion(m):
  aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
  aa_count = []
  for i in aromatic_atoms:
    if i==True:
      aa_count.append(1)
  AromaticAtom = sum(aa_count)
  HeavyAtom = Descriptors.HeavyAtomCount(m)
  AR = AromaticAtom/HeavyAtom
  return AR


def generate_x(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem)
        moldata.append(mol)

    desc_MolLogP=[]
    desc_MolWt=[]
    desc_NumRotatableBonds=[]
    desc_AromaticProportion=[]
    
    #print("len:",len(moldata))
    for mol in moldata:
        desc_MolLogP.append(Descriptors.MolLogP(mol))
        desc_MolWt.append(Descriptors.MolWt(mol))
        desc_NumRotatableBonds.append(Descriptors.NumRotatableBonds(mol))
        desc_AromaticProportion.append(AromaticProportion(mol))

    f={"MolLogP":desc_MolLogP,"MolWt":desc_MolWt,"NumRotatableBonds":desc_NumRotatableBonds,"AromaticProportion":desc_AromaticProportion}
    descriptors = pd.DataFrame.from_dict(f)

    return descriptors

def LogS_to_mg_ml(logs,mw):
    """LogS is directly related to the water solubility of a drug and it is defined as a common solubility
     unit corresponding to the 10-based logarithm of the solubility of a molecule measured in mol/L. 
     The solubility and logS of a drug can be divided in:"""
    mol=10**logs
   # print(mw)
    return str(round(mol/mw*1000,3))+" mg/ml"
    #1 g ----180
    # x g ----1

def mg_ml_to_logS(sol,mw,sol_unit):
    """LogS is directly related to the water solubility of a drug and it is defined as a common solubility
     unit corresponding to the 10-based logarithm of the solubility of a molecule measured in mol/L. 
     The solubility and logS of a drug can be divided in:"""
   #  less than 1 mg/mL at 73° F

    mw=180.1590
    #so mw is g/mol
    #1 g --- 180
    #1 mg --- X
    mol=sol/mw

    LogS=math.log10(mol)
    return LogS

def create_sum(logs,mws):
    f=[]
    for l,m in zip(logs,mws):
        #print(l,m)
        f.append(LogS_to_mg_ml(l,m))
    return f

    
######################
# Page Title
######################


st.write("""
# Molecular Solubility Prediction App
This app predicts the **Solubility (LogS)** values of molecules!
Data obtained from the John S. Delaney. [ESOL:  Estimating Aqueous Solubility Directly from Molecular Structure](https://pubs.acs.org/doi/10.1021/ci034243x). ***J. Chem. Inf. Comput. Sci.*** 2004, 44, 3, 1000-1005.
***
""")


######################
# Input molecules (Side Panel)
######################

st.sidebar.header('User Input Features')

## Read SMILES input
SMILES_input = "NCCCC\nCCC\nCN"

SMILES = st.sidebar.text_area("SMILES input (separate different by new line)", SMILES_input)
#SMILES = "C\n" + SMILES #Adds C as a dummy, first item
SMILES = SMILES.split('\n')
print(SMILES)
st.header('Input SMILES')
#SMILES[1:] # Skips the dummy first item

## Calculate molecular descriptors
st.header('Computed molecular descriptors')
X = generate_x(SMILES)
X
#X[1:] # Skips the dummy first item

######################
# Pre-built model
######################

# Reads in saved model
load_model = pickle.load(open('solubility_model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_model.predict(X)
final_df=pd.DataFrame({"LogS":prediction})
final_df["solubility"]=create_sum(prediction,X.iloc[:,1])

st.header('Predicted Solubility')
#prediction[1:] # Skips the dummy first item
st.dataframe(final_df)

m = Chem.MolFromSmiles(SMILES[0])
im=Draw.MolToImage(m)

st.header('First SMILES Product Visualized')

st.image(im)
