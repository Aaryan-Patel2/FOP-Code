#!/usr/bin/env python3
"""
Multi-Objective Molecular Guidance for GCDM
Combines: Affinity (pKd), Kinetics (k_off), and Synthetic Accessibility (SA)

Novel Contribution: Kinetics-aware drug design (fast dissociation for FOP)
Established Foundation: SA-guided synthesis planning

Based on recent advances in multi-objective molecular optimization
(Nature, 2024) with FOP-specific kinetics constraints.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, QED


class MultiObjectiveScorer:
    """
    Multi-objective molecular scoring for drug discovery.
    
    Objectives:
        1. Affinity (pKd): Target binding strength (7-8 for FOP)
        2. Kinetics (k_off): Fast dissociation (0.1-1 s⁻¹ for FOP)
        3. Synthetic Accessibility (SA): Easy synthesis (1-10 scale, lower=easier)
    
    Weights can be tuned based on discovery phase:
        - Early: High SA weight (synthesize quickly)
        - Mid: Balanced (optimize all)
        - Late: High affinity/kinetics (fine-tune binding)
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        target_pkd: float = 7.5,
        target_koff_min: float = 0.1,
        target_koff_max: float = 1.0,
        target_sa_max: float = 4.0,
        w_affinity: float = 0.4,
        w_kinetics: float = 0.4,
        w_sa: float = 0.2,
        penalty_invalid: float = -10.0
    ):
        """
        Initialize multi-objective scorer.
        
        Args:
            checkpoint_path: Path to Bayesian affinity predictor checkpoint
            target_pkd: Target affinity (pKd units, typically 7-8 for FOP)
            target_koff_min: Minimum acceptable k_off (s⁻¹)
            target_koff_max: Maximum acceptable k_off (s⁻¹)
            target_sa_max: Maximum acceptable SA score (1-10, lower=easier)
            w_affinity: Weight for affinity objective (0-1)
            w_kinetics: Weight for kinetics objective (0-1)
            w_sa: Weight for SA objective (0-1)
            penalty_invalid: Score for invalid/failed molecules
        """
        self.target_pkd = target_pkd
        self.target_koff_min = target_koff_min
        self.target_koff_max = target_koff_max
        self.target_sa_max = target_sa_max
        
        # Normalize weights
        total_weight = w_affinity + w_kinetics + w_sa
        self.w_affinity = w_affinity / total_weight
        self.w_kinetics = w_kinetics / total_weight
        self.w_sa = w_sa / total_weight
        
        self.penalty_invalid = penalty_invalid
        
        # Load affinity predictor
        self._load_predictor(checkpoint_path)
        
        # Load SA scorer
        self._load_sa_scorer()
    
    def _load_predictor(self, checkpoint_path: str):
        """Load Bayesian affinity predictor (includes k_off prediction)"""
        # quick_start.py is in the root directory, not in models/
        import sys
        from pathlib import Path
        root_dir = Path(__file__).parent.parent
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))
        
        from quick_start import AffinityPredictor
        
        try:
            self.predictor = AffinityPredictor(checkpoint_path=checkpoint_path)
            print(f"✓ Loaded affinity predictor from: {checkpoint_path}")
        except Exception as e:
            print(f"ERROR loading predictor: {e}")
            raise
    
    def _load_sa_scorer(self):
        """Load RDKit SA score calculator"""
        try:
            from rdkit.Chem import RDConfig
            import sys
            import os
            sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
            import sascorer
            self.sa_scorer = sascorer
            print("✓ Loaded SA scorer")
        except Exception as e:
            print(f"WARNING: SA scorer not available: {e}")
            self.sa_scorer = None
    
    def calculate_sa_score(self, smiles: str) -> Optional[float]:
        """
        Calculate Synthetic Accessibility Score.
        
        Args:
            smiles: Molecule SMILES string
        
        Returns:
            SA score (1=easy, 10=hard) or None if calculation fails
        """
        if self.sa_scorer is None:
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return self.sa_scorer.calculateScore(mol)
        except Exception as e:
            print(f"SA calculation failed: {e}")
            return None
    
    def predict_affinity_and_kinetics(
        self,
        smiles: str,
        protein_sequence: str
    ) -> Dict[str, float]:
        """
        Predict affinity and kinetics using Bayesian predictor.
        
        Args:
            smiles: Ligand SMILES
            protein_sequence: Target protein sequence
        
        Returns:
            Dictionary with affinity, koff, residence_time, uncertainty
        """
        try:
            result = self.predictor.predict(
                protein_sequence=protein_sequence,
                ligand_smiles=smiles
            )
            return result
        except Exception as e:
            print(f"Prediction failed for {smiles}: {e}")
            return {
                'affinity': 0.0,
                'koff': 0.0,
                'residence_time': 0.0,
                'uncertainty': 999.0
            }
    
    def score_affinity_component(self, pkd: float) -> float:
        """
        Score affinity component (normalized 0-1).
        
        For FOP: Target pKd ~7-8 (moderate affinity)
        Scoring: Gaussian centered at target_pkd with sigma=1.5
        """
        sigma = 1.5
        score = np.exp(-((pkd - self.target_pkd) ** 2) / (2 * sigma ** 2))
        return score
    
    def score_kinetics_component(self, koff: float) -> float:
        """
        Score kinetics component (normalized 0-1).
        
        For FOP: Target k_off in range [0.1, 1.0] s⁻¹ (fast dissociation)
        Scoring: 
            - 1.0 if koff in target range
            - Penalty if too slow or too fast
        """
        if koff <= 0:
            return 0.0
        
        if self.target_koff_min <= koff <= self.target_koff_max:
            # Within target range - perfect score
            return 1.0
        elif koff < self.target_koff_min:
            # Too slow (too tight binding) - penalty
            ratio = koff / self.target_koff_min
            return ratio ** 2  # Quadratic penalty
        else:
            # Too fast (too weak binding) - penalty
            ratio = self.target_koff_max / koff
            return ratio ** 2  # Quadratic penalty
    
    def score_sa_component(self, sa_score: float) -> float:
        """
        Score synthetic accessibility component (normalized 0-1).
        
        SA scale: 1 (easy) to 10 (hard)
        Scoring:
            - 1.0 if SA <= target_sa_max (easy to synthesize)
            - Linear penalty if SA > target_sa_max
        """
        if sa_score is None:
            return 0.5  # Neutral score if SA unavailable
        
        if sa_score <= self.target_sa_max:
            return 1.0
        else:
            # Linear penalty for difficult synthesis
            penalty = (sa_score - self.target_sa_max) / (10.0 - self.target_sa_max)
            return max(0.0, 1.0 - penalty)
    
    def score_molecule(
        self,
        smiles: str,
        protein_sequence: str,
        return_components: bool = True
    ) -> Dict[str, float]:
        """
        Calculate multi-objective score for a molecule.
        
        Args:
            smiles: Molecule SMILES
            protein_sequence: Target protein sequence
            return_components: If True, return breakdown of scores
        
        Returns:
            Dictionary with:
                - total_score: Weighted combination (0-1)
                - affinity_score, kinetics_score, sa_score: Components (0-1)
                - pkd, koff, sa_value: Raw predicted values
                - valid: Boolean flag
        """
        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'total_score': self.penalty_invalid,
                'valid': False,
                'error': 'Invalid SMILES'
            }
        
        # Predict affinity and kinetics
        pred = self.predict_affinity_and_kinetics(smiles, protein_sequence)
        pkd = pred['affinity']
        koff = pred['koff']
        
        # Calculate SA score
        sa_value = self.calculate_sa_score(smiles)
        
        # Score individual components
        affinity_score = self.score_affinity_component(pkd)
        kinetics_score = self.score_kinetics_component(koff)
        sa_score = self.score_sa_component(sa_value)
        
        # Weighted combination
        total_score = (
            self.w_affinity * affinity_score +
            self.w_kinetics * kinetics_score +
            self.w_sa * sa_score
        )
        
        result = {
            'total_score': total_score,
            'valid': True,
            'pkd': pkd,
            'koff': koff,
            'sa_value': sa_value,
            'affinity_score': affinity_score,
            'kinetics_score': kinetics_score,
            'sa_score': sa_score,
            'uncertainty': pred.get('uncertainty', 0.0)
        }
        
        if not return_components:
            # Return only total score for efficiency
            return {'total_score': total_score}
        
        return result
    
    def rank_molecules(
        self,
        smiles_list: list,
        protein_sequence: str,
        top_k: int = 10
    ) -> list:
        """
        Rank molecules by multi-objective score.
        
        Args:
            smiles_list: List of SMILES strings
            protein_sequence: Target protein sequence
            top_k: Number of top molecules to return
        
        Returns:
            List of (smiles, score_dict) tuples, sorted by total_score
        """
        scored_molecules = []
        
        for smiles in smiles_list:
            score_dict = self.score_molecule(smiles, protein_sequence)
            if score_dict['valid']:
                scored_molecules.append((smiles, score_dict))
        
        # Sort by total_score (descending)
        scored_molecules.sort(key=lambda x: x[1]['total_score'], reverse=True)
        
        return scored_molecules[:top_k]
    
    def pareto_frontier(
        self,
        smiles_list: list,
        protein_sequence: str,
        objectives: list = ['affinity_score', 'kinetics_score', 'sa_score']
    ) -> list:
        """
        Find Pareto-optimal molecules (non-dominated solutions).
        
        A molecule is Pareto-optimal if no other molecule is better in all objectives.
        
        Args:
            smiles_list: List of SMILES strings
            protein_sequence: Target protein sequence
            objectives: List of objective names to consider
        
        Returns:
            List of Pareto-optimal (smiles, score_dict) tuples
        """
        # Score all molecules
        scored_molecules = []
        for smiles in smiles_list:
            score_dict = self.score_molecule(smiles, protein_sequence)
            if score_dict['valid']:
                scored_molecules.append((smiles, score_dict))
        
        # Find Pareto frontier
        pareto_set = []
        
        for i, (smiles_i, scores_i) in enumerate(scored_molecules):
            is_dominated = False
            
            for j, (smiles_j, scores_j) in enumerate(scored_molecules):
                if i == j:
                    continue
                
                # Check if j dominates i (better in all objectives)
                dominates = True
                for obj in objectives:
                    if scores_j[obj] <= scores_i[obj]:
                        dominates = False
                        break
                
                if dominates:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_set.append((smiles_i, scores_i))
        
        return pareto_set
    
    def get_weights_summary(self) -> str:
        """Return summary of current objective weights"""
        return (
            f"Multi-Objective Weights:\n"
            f"  Affinity (pKd):     {self.w_affinity:.2f} (target: {self.target_pkd:.1f})\n"
            f"  Kinetics (k_off):   {self.w_kinetics:.2f} (target: {self.target_koff_min:.2f}-{self.target_koff_max:.2f} s⁻¹)\n"
            f"  Synthetic Access.:  {self.w_sa:.2f} (max SA: {self.target_sa_max:.1f})\n"
        )


def main():
    """Example usage of multi-objective scoring"""
    
    # Initialize scorer
    scorer = MultiObjectiveScorer(
        checkpoint_path='trained_models/best_model.ckpt',
        target_pkd=7.5,
        target_koff_min=0.1,
        target_koff_max=1.0,
        target_sa_max=4.0,
        w_affinity=0.4,
        w_kinetics=0.4,
        w_sa=0.2
    )
    
    print(scorer.get_weights_summary())
    
    # Example molecules (from your redocking results)
    test_molecules = {
        'gcdm_46': 'CC1CCC(C2NCCC3C2C2NCNC4NC(N5CCCC5)C3C42)CC1',
        'gcdm_25': 'CCC1CCC(N)CC1=C=CC1NCC2C3CCC(C)OC3CCC2C1C(F)F',
        'reference': 'COc1ccc(cc1)Nc2nccc(n2)c3c[nH]nc3c4cccnc4'
    }
    
    # ACVR1 sequence (example - replace with full sequence)
    protein_seq = "MTEYKLVVVGAGG"  # Placeholder
    
    print("\n" + "="*70)
    print("MULTI-OBJECTIVE SCORING")
    print("="*70)
    
    for mol_id, smiles in test_molecules.items():
        result = scorer.score_molecule(smiles, protein_seq)
        
        print(f"\n{mol_id}:")
        print(f"  Total Score:      {result['total_score']:.3f}")
        print(f"  Affinity (pKd):   {result['pkd']:.2f} → score: {result['affinity_score']:.3f}")
        print(f"  Kinetics (k_off): {result['koff']:.3f} s⁻¹ → score: {result['kinetics_score']:.3f}")
        print(f"  SA Score:         {result['sa_value']:.2f} → score: {result['sa_score']:.3f}")
    
    # Rank molecules
    print("\n" + "="*70)
    print("RANKING BY MULTI-OBJECTIVE SCORE")
    print("="*70)
    
    ranked = scorer.rank_molecules(
        list(test_molecules.values()),
        protein_seq,
        top_k=3
    )
    
    for i, (smiles, scores) in enumerate(ranked, 1):
        print(f"{i}. Score: {scores['total_score']:.3f} | pKd: {scores['pkd']:.2f} | "
              f"k_off: {scores['koff']:.3f} | SA: {scores['sa_value']:.2f}")


if __name__ == "__main__":
    main()
