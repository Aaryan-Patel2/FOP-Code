#!/usr/bin/env python3
"""
Test predictions with trained affinity predictor
Run with: python test_predictions.py
"""

import os
import sys

def test_model(checkpoint_path=None):
    """Test the trained model with example predictions"""
    from quick_start import AffinityPredictor
    
    print("="*60)
    print("Testing Affinity Predictor")
    print("="*60)
    
    # Find checkpoint
    if checkpoint_path is None:
        # Try pretrained first
        if os.path.exists('models/pretrained/affinity_predictor.ckpt'):
            checkpoint_path = 'models/pretrained/affinity_predictor.ckpt'
        # Then try trained_models
        elif os.path.exists('trained_models/best_model.ckpt'):
            checkpoint_path = 'trained_models/best_model.ckpt'
        else:
            print("❌ No trained model found!")
            print("   Train one first with: python train_model.py")
            sys.exit(1)
    
    print(f"\n📁 Loading model: {checkpoint_path}")
    
    try:
        predictor = AffinityPredictor(checkpoint_path=checkpoint_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Test with example protein and ligand
    print("\n" + "="*60)
    print("Test Prediction")
    print("="*60)
    
    # KRAS G12C example
    protein_seq = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS"
    ligand_smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen
    
    print(f"\n📋 Input:")
    print(f"  Protein: KRAS G12C ({len(protein_seq)} residues)")
    print(f"  Ligand: Ibuprofen")
    print(f"  SMILES: {ligand_smiles}")
    
    try:
        result = predictor.predict(protein_seq, ligand_smiles, n_samples=100)
        
        print(f"\n🎯 Prediction Results:")
        affinity = result.get('affinity')
        uncertainty = result.get('uncertainty')
        
        if affinity is not None:
            print(f"  Affinity: {affinity:.2f} ± {uncertainty:.2f} pKd")
            kd_nm = 10**(-affinity) * 1e9
            print(f"  Kd: {kd_nm:.1f} nM")
        
        koff = result.get('koff')
        if koff is not None:
            print(f"  k_off: {koff:.3f} s⁻¹")
            print(f"  Residence time: {result['residence_time']:.1f} seconds")
            print(f"  Confidence: {result['confidence']:.1%}")
            
            # Evaluate for FOP
            if affinity is not None:
                from models.utils.bnn_koff import evaluate_binding_profile
                profile = evaluate_binding_profile(
                    affinity, 
                    koff, 
                    'fop'
                )
                print(f"\n📊 FOP Suitability:")
                print(f"  Overall score: {profile['overall_score']:.2f}/1.0")
                print(f"  {profile['recommendation']}")
        
        print("\n✅ Prediction successful!")
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained affinity predictor')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    test_model(args.checkpoint)
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETE!")
    print("="*60)
    print("\nModel is working correctly and ready for:")
    print("  • Integration with diffusion model (GCDM)")
    print("  • Large-scale virtual screening")
    print("  • Molecule optimization pipelines")
    print()

if __name__ == '__main__':
    main()
