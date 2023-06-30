import os
import shutil
import requests

def store_pdb_structure(pdb_id):
    # Construct the URL to download the PDB file
    cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"

    # Send an HTTP request to download the PDB file
    response_cif = requests.get(cif_url, stream=True)

    # Check if the request was successful
    if response_cif.status_code == 200:
        # Generate a unique temporary file name
        temp_file_cif = f"{pdb_id}.cif"
             
        try:
            # Save the downloaded PDB file to the temporary location
            with open(temp_file_cif, "wb") as file:
                response_cif.raw.decode_content = True
                shutil.copyfileobj(response_cif.raw, file)
            print(f"PDB structure '{pdb_id}' has been temporarily stored as '{temp_file_cif}'")
        except Exception as e:
            return(f"Error occurred while storing the PDB structure: {e}")
        
    else:
        return(f"Error: Unable to download PDB structure with ID '{pdb_id}'")
