import pandas as pd
import os
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

# Configuration
MERGED_DATA = "merged_subsidies_sirene.csv"
OUTPUT_KG = "knowledge_graph.tsv"
MODEL_DIR = "pykeen_model"

def build_base_graph(df):
    triplets = []
    
    for _, row in df.iterrows():
        ent = f"ENTREPRISE_{row['siren']}"
        sec = f"SECTEUR_{row['activitePrincipaleUniteLegale']}"
        file_node = f"FICHIER_{row['source_file']}"
        funder = row['source_funder']
        
        triplets.append([ent, "OPERATES_IN", sec])
        triplets.append([ent, "FUNDED_BY", funder])
        triplets.append([ent, "RECEIVED_SUBSIDY_VIA", file_node])
        
        try:
            amt_str = str(row.get('standard_amount', '0')).replace(' ', '').replace(' ', '').replace(',', '.')
            amt = float(amt_str)
            if amt > 100000:
                amt_node = "BRACKET_ABOVE_100K"
            elif amt > 50000:
                amt_node = "BRACKET_50K_100K"
            else:
                amt_node = "BRACKET_BELOW_50K"
            triplets.append([ent, "RECEIVED_AMOUNT", amt_node])
        except ValueError:
            pass
            
    return pd.DataFrame(triplets, columns=["head", "relation", "tail"])

def apply_logic(df_kg):
    new_triplets = []
    
    # Rule 1: Cross-Ministry/Region Funding
    funders = df_kg[df_kg['relation'] == 'FUNDED_BY'].groupby('head')['tail'].nunique()
    multi_source = funders[funders > 1].index.tolist()
    for ent in multi_source:
        new_triplets.append([ent, "HAS_STATUS", "MULTI_SOURCE_FUNDING"])
        
    # Rule 2: Recurring Beneficiary
    files = df_kg[df_kg['relation'] == 'RECEIVED_SUBSIDY_VIA'].groupby('head')['tail'].nunique()
    recurring = files[files > 1].index.tolist()
    for ent in recurring:
        new_triplets.append([ent, "HAS_STATUS", "RECURRING_BENEFICIARY"])

    # Rule 3: Agri Research Actor
    rd_ents = df_kg[(df_kg['relation'] == 'OPERATES_IN') & (df_kg['tail'] == 'SECTEUR_72.19Z')]['head'].tolist()
    agri_ents = df_kg[(df_kg['relation'] == 'FUNDED_BY') & (df_kg['tail'] == 'MINISTERE_AGRICULTURE')]['head'].tolist()
    valid_rd = set(rd_ents).intersection(set(agri_ents))
    for ent in valid_rd:
        new_triplets.append([ent, "IS_A", "AGRI_RESEARCH_ACTOR"])
        
    # Rule 4: Green Transition Actor
    eco_ents = df_kg[(df_kg['relation'] == 'FUNDED_BY') & (df_kg['tail'] == 'MINISTERE_ECOLOGIE')]['head'].tolist()
    for ent in eco_ents:
        new_triplets.append([ent, "ENGAGED_IN", "GREEN_TRANSITION"])

    # Rule 5: Agriculture Funded
    for ent in agri_ents:
        new_triplets.append([ent, "HAS_STATUS", "AGRICULTURE_FUNDED"])

    # Rule 6: Brittany Region Funded
    bretagne_ents = df_kg[(df_kg['relation'] == 'FUNDED_BY') & (df_kg['tail'] == 'REGION_BRETAGNE')]['head'].tolist()
    for ent in bretagne_ents:
        new_triplets.append([ent, "HAS_STATUS", "BRITTANY_FUNDED"])

    if new_triplets:
        df_inf = pd.DataFrame(new_triplets, columns=["head", "relation", "tail"])
        return pd.concat([df_kg, df_inf], ignore_index=True)
        
    return df_kg

def train_and_save_kge(kg_path):
    tf = TriplesFactory.from_path(kg_path)
    result = pipeline(
        training=tf,
        testing=tf, 
        model='TransE',
        training_kwargs=dict(num_epochs=50),
        random_seed=42,
        device='cpu'
    )
    result.save_to_directory(MODEL_DIR)

def main():
    print("1. Loading merged dataset...")
    if not os.path.exists(MERGED_DATA):
        print(f"Error: '{MERGED_DATA}' not found.")
        return

    df = pd.read_csv(MERGED_DATA, dtype=str)
    
    print("2. Building and enriching Knowledge Graph...")
    df_kg = build_base_graph(df)
    df_kg_enriched = apply_logic(df_kg)
    
    df_kg_enriched.drop_duplicates(inplace=True)
    df_kg_enriched.to_csv(OUTPUT_KG, sep='\t', index=False)
    print(f"Graph saved to '{OUTPUT_KG}' ({len(df_kg_enriched)} triplets).")
    
    print("\n3. Training PyKEEN embedding model (This will take 3 to 10 minutes)...")
    train_and_save_kge(OUTPUT_KG)
    print(f"Model successfully saved to directory '{MODEL_DIR}'. Pipeline Phase 2 complete.")

if __name__ == "__main__":
    main()