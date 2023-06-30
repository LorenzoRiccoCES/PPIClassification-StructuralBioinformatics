from SupportScripts import MLPFunctions, getPDB
import subprocess
import os
import torch
import pandas as pd
import numpy as np


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

    pdb_id = input('Enter the PDB ID: ')
    PDBresult = getPDB.store_pdb_structure(pdb_id)
    # in case of error
    if (PDBresult != None):
        terminazione = ': execution finished'
        outputOperation = f'{PDBresult}{terminazione}'
        print(outputOperation)
    else:
        #load .tsv 
        data = []
        #run calc_features.py
        execute_program(f'contacts_classification/calc_features.py', pdb_id)
        #table extraction from .tsv file
        data = generateFiltersTSV(f"outputFolder/{pdb_id}.tsv")

        # Convert input data from DataFrame to PyTorch tensor
        data.dropna(inplace=True)     
        data = MLPFunctions.encode_data(data)
        data = data.reset_index(drop=True)
        shape = data.shape[1]
        #crated table for prediction output
        df = pd.DataFrame(0, columns=['HBOND', 'IONIC', 'PICATION', 'PIPISTACK', 'SSBOND', 'VDW'], index=range(data.shape[0]))
        
        Data_tensor = torch.tensor(data.to_numpy())      
        Data_tensor = Data_tensor.to(torch.float32)

        #model/s load
        models = []
        models = load_model(f'models/', models)

        qtaModelli = len(models)

        #check if there is at least 1 model
        if qtaModelli < 1:
            print("No models were found in models folder: Execution terminated")

        else:
            # Iterate over the files and display their names and numbers
            file_list = os.listdir("models/")
            # Iterate over the files and display their names and numbers
            print("Choose which model to use: ")
            for i, file_name in enumerate(file_list):
                print(f"{i}: {file_name}")
            #select number in list corresponding to desired model
            selected_number = input("Select a number from the list:")
            #error handling
            try:
                selected_number = int(selected_number)
                if selected_number < 0 or selected_number >= len(file_list):
                    raise ValueError
            except ValueError:
                print("Selected model number not present: execution terminated.")
            else:
                #generate prediction from input data and model
                for model in models:
                    model.to(device)
                Data_tensor = Data_tensor.to(device)
                '''
                test = models[selected_number]
                output = test(Data_tensor)
                #print(output)
                y_pred = []
                threshold = 0.5
                y_pred.append(output[:, 1] > threshold)
                y_pred = torch.stack(y_pred, dim=1)

                # Convert predicted labels to binary indicator matrix
                y_pred_bin = np.zeros(shape=[data.shape[0], data.shape[0]])
                for i in range(y_pred.shape[0]):
                    y_pred_bin[i, y_pred[i]] = 1
                #print(y_pred_bin)

                # Convert predicted probabilities to numpy array
                y_pred_prob = output.cpu().detach().numpy()

                # Print the predicted probabilities for each row and label
                
                for i in range(y_pred_prob.shape[0]):
                    print(f'Row {i}:')
                    for j in range(y_pred_prob.shape[1]):
                        print(f'Label {j}: Predicted probability = {y_pred_prob[i,j]:.4f}')
                        '''
                with torch.no_grad():
                    model = models[selected_number]
                    model.eval()
                    predictions = model.forward(Data_tensor)
                    predicted_probabilities = torch.nn.functional.softmax(predictions, dim=1)
                    _, predicted_labels = torch.max(predicted_probabilities, dim=1)
                    # Print the predicted probabilities and labels for each row
                    for i in range(len(predicted_probabilities)):
                        #print(f"Row {i + 1}: Probabilities: {predicted_probabilities[i]}")
                        print(f"Row {i + 1}: Predicted Label: {predicted_labels[i]}")