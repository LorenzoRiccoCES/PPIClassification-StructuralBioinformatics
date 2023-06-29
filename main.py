from SupportScripts import MLPFunctions, getPDB
import subprocess
import os
import torch
import pandas as pd
#rimuovere
from torch import nn



def execute_program(program_path, pdb_id):
    try:
        subprocess.run(["python3", program_path, f"{pdb_id}.cif", "-out_dir", "outputFolder"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing the program: {e}")
    
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

'''
def load_model(folder_path, shape, models = []):
    #for cycle: iterate all file in folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # Check if the current item is a file
        if os.path.isfile(file_path):
            #load model
            loaded_model = MLPFunctions.MLP(shape,10)
            loaded_model.load_state_dict(torch.load(file_path))
            #loaded_model.eval()
            models.append(loaded_model) 
    return models
'''

def load_model(folder_path, models=[]):
    # shape = 20  # Set the input size and hidden size as required
    hidden = 50
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            loaded_model = MLPFunctions.MLP(shape, hidden)  # Pass the correct input size and hidden size
            loaded_model.load_state_dict(torch.load(file_path))
            models.append(loaded_model)
    return models


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []

    pdb_id = input('Enter the PDB ID: ')
    PDBresult = getPDB.store_pdb_structure(pdb_id)
    # in case of error
    if (PDBresult != None):
        terminazione = ': esecuzione terminata'
        outputOperation = f'{PDBresult}{terminazione}'
        print(outputOperation)
    else:
        #load .tsv 
        data = []
        execute_program(f'contacts_classification/calc_features.py', pdb_id)
        data = generateFiltersTSV(f"outputFolder/{pdb_id}.tsv")
        #data = data.rank(pct=True).round(1).astype('category')
        print(data)
        print(data.shape[0], " x ", data.shape[1])
        # Convert input data from DataFrame to PyTorch tensor
        data.dropna(inplace=True)
        
        data = MLPFunctions.encode_data(data)

        #print(data)
        data = data.reset_index(drop=True)
        print(data.shape[0], " x ", data.shape[1])
        #data = MLPFunctions.ProteinDataset(data)

        Data_tensor = torch.tensor(data.to_numpy())
        print(data.shape[0], " x ", data.shape[1])
        shape = data.shape[1]
        #Data_tensor = torch.stack([data[i] for i in range(len(data))])

        Data_tensor = Data_tensor.to(torch.float32)
        models = []
        models = load_model(f'models/', models)
        for model in models:
            model.to(device)
        Data_tensor = Data_tensor.to(device)
        test = models[0]
        output = models[0](Data_tensor)
        #print(output)




''' 
problemi da risolvere
    - controllare se posizione di load_model e tutte sottofunzioni sono nel posto corretto
'''