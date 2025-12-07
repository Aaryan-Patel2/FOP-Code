#!/usr/bin/env python3
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from models.bayesian_affinity_predictor import HybridBayesianAffinityNetwork
from models.pdbbind_data_preparation import ProteinSequenceEncoder, SMILESEncoder
from models.utils.bnn_koff import predict_koff_empirical

print('Loading model...')
checkpoint = torch.load('trained_models/best_model-v1.ckpt', map_location='cpu')
ligand_vocab_size = checkpoint['state_dict']['model.ligand_encoder.embedding.weight'].shape[0]
protein_vocab_size = checkpoint['state_dict']['model.protein_encoder.embedding.weight'].shape[0]

model = HybridBayesianAffinityNetwork(
    protein_vocab_size=protein_vocab_size,
    ligand_vocab_size=ligand_vocab_size,
    complex_descriptor_dim=200,
    protein_output_dim=256,
    ligand_output_dim=256,
    complex_output_dim=128,
    fusion_hidden_dims=[512, 256, 128],
    dropout=0.3,
    prior_sigma=1.0
)
state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict, strict=False)
model.eval()

protein_encoder = ProteinSequenceEncoder()
ligand_encoder = SMILESEncoder()
acvr1_seq = 'MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPARTVATRQVVEKEKRATLLLFSNPPNPNNAKKKMEEWTFLRLSQDSRPPNPSLLHGSSPPPPSHRQFPEEESPGDASSSSSSTQSSSDLQAFQTNPSAALVAGSSPTLSGTPSPTGLVTPSSHTVSSPVPPPAPSGGGAEVESAPAGAVGPSSPLPASQPVGGMPDVSPGSAYAVSGSSVFPSSSHVGMGFPAAAGFPFVPSSS'

mols = [m for m in Chem.SDMolSupplier('generated_molecules/acvr1_mutant_3mtf_mol.sdf') if m is not None]
print(f'\nScoring {len(mols)} molecules...\n' + '='*85)

results = []
for i, mol in enumerate(mols, 1):
    smiles = Chem.MolToSmiles(mol)
    protein_tokens = protein_encoder.encode(acvr1_seq, max_length=512)
    ligand_tokens = ligand_encoder.encode(smiles, max_length=128)
    complex_desc = np.zeros(200, dtype=np.float32)
    
    protein_tensor = torch.LongTensor(protein_tokens).unsqueeze(0)
    ligand_tensor = torch.LongTensor(ligand_tokens).unsqueeze(0)
    complex_tensor = torch.FloatTensor(complex_desc).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        pred_affinity, uncertainty = model.predict_with_uncertainty(protein_tensor, ligand_tensor, complex_tensor, n_samples=50)
    
    affinity = pred_affinity.item()
    unc = uncertainty.item()
    koff_result = predict_koff_empirical(affinity)
    kd_nm = 10**(9 - affinity)
    
    results.append((i, smiles, mol.GetNumAtoms(), Descriptors.MolWt(mol), affinity, unc, koff_result.koff_mean, koff_result.residence_time))
    
    print(f"\nMolecule {i}: {mol.GetNumAtoms()} atoms, MW={Descriptors.MolWt(mol):.1f} Da")
    print(f"  Affinity: pKd = {affinity:.2f} ± {unc:.2f} (Kd ~ {kd_nm:.0f} nM)")
    print(f"  Kinetics: k_off = {koff_result.koff_mean:.3f} s⁻¹, τ = {koff_result.residence_time:.1f} s")
    print(f"  SMILES: {smiles}")

print('\n' + '='*85)
print('TOP CANDIDATES FOR FOP THERAPY')
print('='*85)
print('Target: pKd ~7.5 (Kd ~32 nM), k_off ~0.3 s⁻¹')
print('='*85 + '\n')

fop_scores = [(10 - abs(aff-7.5)**2 - 3*abs(koff-0.3)**2, mol_id, smiles, atoms, mw, aff, koff, tau) 
              for mol_id, smiles, atoms, mw, aff, unc, koff, tau in results]
fop_scores.sort(reverse=True)

for rank, (score, mol_id, smiles, atoms, mw, aff, koff, tau) in enumerate(fop_scores, 1):
    kd_nm = 10**(9 - aff)
    print(f"{rank}. Molecule {mol_id} — FOP Score: {score:.2f}")
    print(f"   {atoms} atoms, MW={mw:.1f} Da | pKd={aff:.2f} (Kd~{kd_nm:.0f} nM) | k_off={koff:.3f} s⁻¹, τ={tau:.1f}s")
    print(f"   {smiles}\n")

print('='*85)
print(f'DONE! Scored all {len(results)} molecules.')
print('='*85)
