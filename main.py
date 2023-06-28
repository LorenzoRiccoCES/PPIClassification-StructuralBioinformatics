from SupportScripts import getPDB
import subprocess
import os
import joblib
import torch
import csv
import pandas as pd
#rimuovere
from torch import nn
import json



def execute_program(program_path, pdb_id):
    try:
        subprocess.run(["python3", program_path, f"{pdb_id}.cif", "-out_dir", "outputFolder"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing the program: {e}")

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        return out
    
def generateFiltersTSV(file_path):
    with open(file_path, 'r', encoding='latin-1') as file:
        lines = file.readlines()
    matrix = [line.replace(' ', '').replace('"', '').strip().split('\t') for line in lines]

    df=pd.DataFrame(matrix)
    # Drop the first column
    df = df.drop(df.columns[0], axis=1)
    # drop not used columns
    remove = ["s_ch", "s_resi", "s_ins", "s_resn", "s_ss8", "s_ss3", "t_ch", "t_resi", "t_ins", "t_resn", "t_ss8", "t_ss3"]
    #Drop columns where the values in the first row match any of the strings in 'remove'
    df = df.loc[:, ~df.iloc[0].isin(remove)]
    #find and delete columns with no values
    findVoid = df.iloc[1].str[0].isnull()
    cols_to_drop = df.columns[findVoid]
    df = df.drop(cols_to_drop, axis=1)
    return df


def load_model(folder_path, models = []):
    #for cycle: iterate all file in folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # Check if the current item is a file
        if os.path.isfile(file_path):
            #load .tsv 
            data = []
            data = generateFiltersTSV(file_path)
            #load model
            loaded_model = MLP(data.shape[1],1)
            loaded_model.load_state_dict(torch.load(file_path))
            loaded_model.eval()
            
            models.append(loaded_model)  

if __name__ == "__main__":

    models = []

    pdb_id = input('Enter the PDB ID: ')
    PDBresult = getPDB.store_pdb_structure(pdb_id)
    # in case of error
    if (PDBresult != None):
        terminazione = ': esecuzione terminata'
        outputOperation = f'{PDBresult}{terminazione}'
        print(outputOperation)
    else:
        execute_program(f'contacts_classification/calc_features.py', pdb_id)
        models = []
        models = load_model(f'models/', models)



''' 
problemi da risolvere
    - spostare e far funzionare MLP
    - controllare se posizione di load_model e tutte sottofunzioni sono nel posto corretto

'''