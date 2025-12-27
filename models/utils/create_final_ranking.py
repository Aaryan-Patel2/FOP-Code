#!/usr/bin/env python3
"""
Create final ranked JSON summary combining all molecular properties
"""

import pandas as pd
import json

# Read the docked results
docked_df = pd.read_csv('all_molecules_docked.csv')

# Read the original combined data for full properties
combined_df = pd.read_csv('all_molecules_combined.csv')

# Merge them on ID and SMILES
merged_df = pd.merge(
    docked_df,
    combined_df,
    on=['ID', 'SMILES'],
    how='left'
)

# Calculate a composite FOP suitability score
# Higher is better: balance good affinity, fast k_off, and strong Vina binding
# Score = (Vina_Affinity strength) + (k_off preference) - (overly strong binding penalty)
merged_df['FOP_Score'] = (
    -merged_df['Vina_Affinity_kcal_mol'] * 0.5 +  # Stronger Vina = better
    merged_df['k_off'] * 10 +  # Faster k_off = better
    merged_df['QED'] * 2  # Drug-likeness
)

# Sort by FOP score (higher is better)
merged_df = merged_df.sort_values('FOP_Score', ascending=False)

# Create the final JSON structure
results = {
    "summary": {
        "total_molecules": len(merged_df),
        "good_agreement": len(merged_df[merged_df['Agreement'] == 'Good']),
        "poor_agreement": len(merged_df[merged_df['Agreement'] == 'Poor']),
        "average_vina_affinity": float(merged_df['Vina_Affinity_kcal_mol'].mean()),
        "average_predicted_pkd": float(merged_df['Predicted_pKd'].mean()),
        "average_koff": float(merged_df['k_off'].mean())
    },
    "molecules": []
}

# Add each molecule
for idx, row in merged_df.iterrows():
    molecule = {
        "rank": len(results["molecules"]) + 1,
        "id": row['ID'],
        "smiles": row['SMILES'],
        "fop_suitability_score": round(float(row['FOP_Score']), 3),
        "predicted_affinity": {
            "pKd": round(float(row['Predicted_pKd']), 2),
            "uncertainty": round(float(row['Uncertainty']), 3)
        },
        "kinetics": {
            "k_off_s-1": round(float(row['k_off']), 3),
            "residence_time_s": round(float(row['Residence_Time']), 1)
        },
        "docking": {
            "vina_affinity_kcal_mol": round(float(row['Vina_Affinity_kcal_mol']), 2),
            "vina_pKd": round(float(row['Vina_pKd']), 2),
            "pkd_difference": round(float(row['pKd_Difference']), 2),
            "agreement": row['Agreement']
        },
        "properties": {
            "molecular_weight": round(float(row['MW']), 1),
            "qed": round(float(row['QED']), 3),
            "logP": round(float(row['LogP']), 2)
        }
    }
    results["molecules"].append(molecule)

# Save to JSON
with open('final_ranked_molecules.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Created final_ranked_molecules.json with {len(results['molecules'])} molecules")
print(f"\nTop 5 by FOP Suitability Score:")
for mol in results["molecules"][:5]:
    print(f"  {mol['rank']}. {mol['id']}: Score={mol['fop_suitability_score']:.2f}, "
          f"Vina={mol['docking']['vina_affinity_kcal_mol']} kcal/mol, "
          f"k_off={mol['kinetics']['k_off_s-1']} s⁻¹")
