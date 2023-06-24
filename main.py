from SupportScripts import getPDB
import subprocess

# Load the model
#model = tf.keras.models.load_model(r"C:\Users\emanu\Desktop\StructBio\model.h5", compile=False)

def execute_program(program_path,list=[]):
    try:
        subprocess.run([program_path, "temp.cif", "output/"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing the program: {e}")



def main():
    
    pdb_id = input("Enter the PDB ID: ")
    getPDB.store_pdb_structure(pdb_id)

    #for execution of calc_features.py: $python3 calc_features.py my_pdb.cif -out_dir output/
    execute_program("contacts_classification/calc_features.py")



if __name__ == "__main__":
    main()
