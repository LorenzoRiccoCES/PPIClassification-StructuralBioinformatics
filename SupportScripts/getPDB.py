import os
import shutil
import requests

def store_pdb_structure(pdb_id):
    # Construct the URL to download the PDB file
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"

    # Send an HTTP request to download the PDB file
    response = requests.get(url, stream=True)
    response_cif = requests.get(cif_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Generate a unique temporary file name
        temp_file = "temp.pdb"
        temp_file_cif = "temp.cif"      
        
        try:
            # Save the downloaded PDB file to the temporary location
            with open(temp_file, "wb") as file:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, file)
            print(f"PDB structure '{pdb_id}' has been temporarily stored as '{temp_file}' and '{temp_file_cif}' .")
        except Exception as e:
            print(f"Error occurred while storing the PDB structure: {e}")

        # Save the CIF file to the local directory
        with open(temp_file_cif, "wb") as file:
            file.write(response.content)
    else:
        print(f"Error: Unable to download PDB structure with ID '{pdb_id}'.")
