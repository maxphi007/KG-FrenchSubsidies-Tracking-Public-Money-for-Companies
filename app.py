import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import json
from pykeen.triples import TriplesFactory

# Configuration paths
MODEL_PATH = "pykeen_model/trained_model.pkl"
KG_PATH = "knowledge_graph.tsv"
DICT_PATH = "siren_name_mapping.json"
MERGED_DATA = "merged_subsidies_sirene.csv"

st.set_page_config(page_title="Agri-Subsidies AI Explorer", layout="wide")

@st.cache_resource
def load_ai_engine():
    tf = TriplesFactory.from_path(KG_PATH)
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    embeddings = model.entity_representations[0](indices=None).detach()
    return tf, embeddings

@st.cache_data
def load_metadata():
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        siren_dict = json.load(f)
    df_kg = pd.read_csv(KG_PATH, sep='\t')
    df_merged = pd.read_csv(MERGED_DATA, dtype=str)
    return siren_dict, df_kg, df_merged

try:
    tf, embeddings = load_ai_engine()
    siren_dict, df_kg, df_merged = load_metadata()
except Exception as e:
    st.error(f"Error loading files. Please ensure Phase 1 and 2 scripts ran perfectly. Details: {e}")
    st.stop()

entity_to_id = tf.entity_to_id
id_to_entity = {v: k for k, v in entity_to_id.items()}

# Base dictionary of all available companies
all_companies = {
    name: siren for siren, name in siren_dict.items() 
    if f"ENTREPRISE_{siren}" in entity_to_id
}

# --- NEW: SIDEBAR FOR FILTERING ---
st.sidebar.header("🔍 Filters & Categories")

# 1. Dynamically extract all logical labels from the Knowledge Graph
logical_relations = ['HAS_STATUS', 'IS_A', 'ENGAGED_IN']
raw_labels = df_kg[df_kg['relation'].isin(logical_relations)]['tail'].unique().tolist()
clean_labels = sorted([label.replace('_', ' ') for label in raw_labels])

# 2. Filter selection
selected_filter = st.sidebar.selectbox(
    "Filter companies by Label:", 
    options=["All Companies"] + clean_labels
)

st.sidebar.markdown("---")
st.sidebar.info("This filter reduces the list of companies in the main search bar to those possessing the selected status.")

# 3. Apply the filter to the company list
filtered_companies = {}
if selected_filter != "All Companies":
    raw_target_label = selected_filter.replace(' ', '_')
    # Find all entities holding this specific label
    matching_entities = df_kg[(df_kg['relation'].isin(logical_relations)) & (df_kg['tail'] == raw_target_label)]['head'].tolist()
    
    # Keep only companies that are in the matching entities list
    for name, siren in all_companies.items():
        if f"ENTREPRISE_{siren}" in matching_entities:
            filtered_companies[name] = siren
else:
    filtered_companies = all_companies.copy()

# --- MAIN UI ---
st.title("🌾 Agri-Subsidies AI Explorer")
st.markdown("Explore similar companies based on multi-ministry funding patterns using Knowledge Graph Embeddings.")

# Show how many companies match the filter
st.caption(f"Currently showing {len(filtered_companies)} companies out of {len(all_companies)}.")

# Search bar (now uses the filtered list)
selected_name = st.selectbox(
    "Search for a subsidized company:", 
    options=sorted(filtered_companies.keys()),
    index=None,
    placeholder="Start typing a company name..." if filtered_companies else "No companies match this filter."
)

if selected_name:
    st.markdown("---")
    siren = filtered_companies[selected_name]
    kg_entity = f"ENTREPRISE_{siren}"
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🏢 Company Profile")
        st.write(f"**Name:** {selected_name}")
        st.write(f"**SIREN:** {siren}")
        
        # Calculate Exact Total Amount
        company_data = df_merged[df_merged['siren'] == siren]
        total_amount = 0.0
        for amt_str in company_data['standard_amount'].dropna():
            clean_amt = str(amt_str).replace(' ', '').replace(' ', '').replace(',', '.')
            try:
                total_amount += float(clean_amt)
            except ValueError:
                pass
                
        st.metric(label="Total Subsidies Received", value=f"{total_amount:,.2f} €".replace(',', ' '))
        
        # Display inferred logical rules
        company_facts = df_kg[df_kg['head'] == kg_entity]
        st.write("**Logical Labels (Inferred):**")
        found_labels = False
        for _, row in company_facts.iterrows():
            if row['relation'] in logical_relations:
                st.info(row['tail'].replace('_', ' '))
                found_labels = True
        if not found_labels:
            st.write("*No special logical label inferred for this company.*")

    with col2:
        st.subheader("Most Similar companies (Top 5)")
        
        target_id = entity_to_id[kg_entity]
        target_vector = embeddings[target_id]
        similarities = F.cosine_similarity(target_vector.unsqueeze(0), embeddings)
        
        top_k = 6 
        top_scores, top_indices = torch.topk(similarities, top_k)
        
        results = []
        for score, idx in zip(top_scores, top_indices):
            idx_val = idx.item()
            ent_name = id_to_entity[idx_val]
            
            if ent_name.startswith("ENTREPRISE_") and ent_name != kg_entity:
                sim_siren = ent_name.replace("ENTREPRISE_", "")
                real_name = siren_dict.get(sim_siren, "Unknown Name")
                results.append({
                    "Similarity Score": f"{score.item():.2%}",
                    "Company": real_name,
                    "SIREN": sim_siren
                })
        
        if results:
            st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)
        else:
            st.write("No similar companies found.")