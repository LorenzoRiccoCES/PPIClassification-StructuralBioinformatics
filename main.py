from SupportScripts import MLPFunctions, getPDB
import subprocess
import os
import torch
import pandas as pd
import numpy as np
import re


def execute_program(program_path, pdb_id):
    try:
        subprocess.run(["python3", program_path, f"{pdb_id}.cif", "-out_dir", "outputFolder"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing the program: {e}")

def extractInfoFromFile(file_name):
    # Extract number of features
    features = int(re.search(r"_features_°(\d+)", file_name).group(1))

    # Extract columns as a list of integers
    columns_str = re.search(r"_columns_(#[\d-]+)", file_name).group(1)
    columns = [int(col) for col in columns_str[1:].split('-')]
    return features, columns

def generateFiltersTSV(matrix, columns):
    df=pd.DataFrame(matrix)
    # Drop the first column
    df = df.drop(df.columns[0], axis=1)
    # drop not used columns
    #Drop columns not in 'columns'
    df = df.iloc[:, columns]
    df = df.rank(pct=True).round(1)
    return df

def get_files_list(directory):
    files_list = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            files_list.append(file_path)
    return files_list

def load_model(file_path, features):
    hidden=50
    # Pass the correct input size and hidden size
    loaded_model = MLPFunctions.MLP(features, hidden)  
    loaded_model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
    return loaded_model

def check_subfolder_files(folder_path):
    for subfolder_name in os.listdir(folder_path):  # Iterate over subfolders
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):  # Check if it's a subfolder
            files = os.listdir(subfolder_path)
            if len(files) == 0:
                return False
    return True


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
        #check if there is at least 1 model in every subfolder of model
        if not check_subfolder_files("models/"):
            print("At least one subfolder of 'models' does not have any files: Execution terminated")

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
                #run calc_features.py
                execute_program(f'contacts_classification/calc_features.py', pdb_id)

                files = get_files_list(f"models/{file_list[selected_number]}/")
                #order elements in files
                files = sorted(files, key=lambda x: int(x.split("model_")[1].split("_")[0]))

                #load file .tsv
                with open(f"outputFolder/{pdb_id}.tsv", 'r', encoding='latin-1') as file:
                    lines = file.readlines()
                matrix = [line.replace(' ', '').replace('"', '').strip().split('\t') for line in lines]
                
                labels = ['HBOND', 'IONIC', 'PICATION', 'PIPISTACK', 'SSBOND', 'VDW']
                
                iteratore = 0

                all_y_pred = []
                all_y_prob = []

                # Iterate over the files in the selected model's subfolder
                for i, element in enumerate(files):
                    print(labels[i])
                    # Extract the number of features and columns from the file name
                    features, columns = extractInfoFromFile(element)

                    # Load the TSV file and generate a filtered TSV file
                    temp_matrix = matrix
                    data = generateFiltersTSV(temp_matrix, columns)

                    # Convert the filtered data into a PyTorch tensor
                    data.dropna(inplace=True)
                    data = MLPFunctions.encode_data(data)
                    data = data.reset_index(drop=True)
                    Data_tensor = torch.tensor(data.to_numpy())
                    Data_tensor = Data_tensor.to(torch.float32)

                    # Load the model
                    model = load_model(f"{element}", features)

                    # Apply the model to the input data
                    outputs = model(Data_tensor)

                    # Get the predicted probabilities for the positive class
                    probas = outputs[:, 1]

                    threshold = 0.5

                    # Apply the threshold to get the predicted labels
                    predicted_labels = (probas > threshold).float()

                    # Store the predicted labels and probabilities for the current model
                    all_y_pred.append(predicted_labels)
                    all_y_prob.append(probas.detach())

                    # Convert the predicted labels and probabilities to NumPy arrays
                    all_y_pred_np = np.stack(all_y_pred, axis=1)
                    all_y_prob_np = np.stack(all_y_prob, axis=1)

                    # Print the predictions for all models side by side
                    print('Predictions for all models:')
                    for i in range(all_y_pred_np.shape[0]):
                        print(f'Example {i+1}:')
                        print(f'  Predicted labels: {all_y_pred_np[i]}')
                        print(f'  Predicted probabilities: {all_y_prob_np[i]}')