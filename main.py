from SupportScripts import getPDB
import subprocess
import os
# Load the model
#model = tf.keras.models.load_model(r"C:\Users\emanu\Desktop\StructBio\model.h5", compile=False)

def execute_program(program_path, pdb_id):
    try:
        subprocess.run(["python3", program_path, f"{pdb_id}.cif", "-out_dir", "outputFolder"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing the program: {e}")


if __name__ == "__main__":
    pdb_id = input("Enter the PDB ID: ")
    PDBresult = getPDB.store_pdb_structure(pdb_id)
    if (PDBresult != None):
        terminazione = ": esecuzione terminata"
        outputOperation = f"{PDBresult}{terminazione}"
        print(outputOperation)
    else:
        #for execution of calc_features.py: $python3 calc_features.py my_pdb.cif -out_dir output/
        #execute_program("contacts_classification/calc_features.py")
        #full path
        execute_program(f"contacts_classification/calc_features.py", pdb_id)