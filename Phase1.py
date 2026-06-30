import pandas as pd
import glob
import os
import json

# Configuration
BASE_DIR = "data_subsidies"
SIRENE_FILE = "StockUniteLegale.csv"
OUTPUT_MERGED = "merged_subsidies_sirene.csv"
OUTPUT_DICT = "siren_name_mapping.json"

# Source-specific configurations with fallback column names
MAPPINGS = {
    "agriculture": {
        "id_cols": ["Identification du bénéficiaire*"],
        "name_cols": ["Nom du bénéficiaire*"],
        "amt_cols": ["Montant total de la subvention*"],
        "funder": "MINISTERE_AGRICULTURE",
        "separator": ";"
    },
    "ecology": {
        "id_cols": ["siret_beneficiaire", "siren", "siret"],
        "name_cols": ["raison_sociale_beneficiaire", "nom_beneficiaire", "nom_beneficiaire_principal"],
        "amt_cols": ["montant_engage", "montant"],
        "funder": "MINISTERE_ECOLOGIE",
        "separator": ","
    },
    "region": {
        "id_cols": ["idbeneficiaire", "siret", "siren"],
        "name_cols": ["nombeneficiaire", "nom_beneficiaire", "beneficiaire", "raison_sociale", "nom attributaire*"],
        "amt_cols": ["montant", "montant_subvention", "montant_alloue"],
        "funder": "REGION_BRETAGNE",
        "separator": ";" # Adjust to "," if the file uses commas
    }
}

def find_column(df, possible_names):
    # Convert dataframe columns to lowercase for easier matching
    lower_cols = {col.lower(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in lower_cols:
            return lower_cols[name.lower()]
    return None

def main():
    print("1. Loading and cleaning subsidy files by source...")
    df_list = []
    
    for source_folder, config in MAPPINGS.items():
        folder_path = os.path.join(BASE_DIR, source_folder, "*.csv")
        files = glob.glob(folder_path)
        
        for file in files:
            try:
                skip = 1 if source_folder == "agriculture" else 0
                sep = config["separator"]
                
                try:
                    df_temp = pd.read_csv(file, sep=sep, skiprows=skip, dtype=str, encoding='utf-8')
                except UnicodeDecodeError:
                    df_temp = pd.read_csv(file, sep=sep, skiprows=skip, dtype=str, encoding='latin-1')
                
                actual_id_col = find_column(df_temp, config["id_cols"])
                actual_name_col = find_column(df_temp, config["name_cols"])
                actual_amt_col = find_column(df_temp, config["amt_cols"])
                
                if actual_id_col and actual_name_col and actual_amt_col:
                    df_temp['siren'] = df_temp[actual_id_col].fillna('').str.replace(r'\s+', '', regex=True).str[:9]
                    
                    df_temp['standard_name'] = df_temp[actual_name_col]
                    df_temp['standard_amount'] = df_temp[actual_amt_col]
                    df_temp['source_funder'] = config["funder"]
                    df_temp['source_file'] = os.path.basename(file)
                    
                    df_list.append(df_temp)
                else:
                    print(f"Warning: Missing required columns in {os.path.basename(file)}.")
                    print(f"  -> Found ID: {actual_id_col}, Name: {actual_name_col}, Amount: {actual_amt_col}")
                    
            except Exception as e:
                print(f"Error reading {file}: {e}")

    if not df_list:
        print("No valid subsidy data found. Exiting.")
        return

    df_all_subs = pd.concat(df_list, ignore_index=True)
    df_all_subs = df_all_subs[df_all_subs['siren'] != '']
    subsidized_sirens = set(df_all_subs['siren'])
    print(f"Found {len(subsidized_sirens)} unique subsidized SIRENs across all ministries/regions.")

    print("\n2. Creating and saving SIREN-to-Name dictionary...")
    siren_dict = df_all_subs.dropna(subset=['standard_name']).drop_duplicates(subset=['siren']).set_index('siren')['standard_name'].to_dict()
    
    with open(OUTPUT_DICT, 'w', encoding='utf-8') as f:
        json.dump(siren_dict, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(siren_dict)} names to '{OUTPUT_DICT}'.")

    print("\n3. Processing Sirene data in chunks...")
    chunk_size = 100000
    filtered_sirene_list = []

    for chunk in pd.read_csv(SIRENE_FILE, chunksize=chunk_size, dtype=str):
        # Kept 10 and 11 as you mentioned trying it before
        is_agri = chunk['activitePrincipaleUniteLegale'].fillna('').str.startswith(('01', '02', '10', '11'))
        in_subsidy = chunk['siren'].isin(subsidized_sirens)
        
        filtered_chunk = chunk[is_agri | in_subsidy]
        if not filtered_chunk.empty:
            filtered_sirene_list.append(filtered_chunk)

    df_sirene_filtered = pd.concat(filtered_sirene_list, ignore_index=True)
    cols_to_keep = ['siren', 'nomUniteLegale', 'activitePrincipaleUniteLegale']
    df_sirene_filtered = df_sirene_filtered[cols_to_keep]

    print("\n4. Merging datasets...")
    df_merged = pd.merge(df_all_subs, df_sirene_filtered, on='siren', how='inner')
    print(f"Merge successful! Total records: {len(df_merged)}")

    final_cols = ['siren', 'standard_name', 'standard_amount', 'source_funder', 'source_file', 'activitePrincipaleUniteLegale']
    df_merged[final_cols].to_csv(OUTPUT_MERGED, index=False)
    print(f"Data saved to '{OUTPUT_MERGED}'. Pipeline Phase 1 complete.")

if __name__ == "__main__":
    main()