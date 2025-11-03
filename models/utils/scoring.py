# In your scoring.py, add bounds checking:
def calculate_temporary_inhibition_score(mutant_score, wt_score):
    """With sanity checks for impossible scores"""
    
    # SANITY CHECK: Docking scores should be between -15 and 0
    if mutant_score < -15 or mutant_score > 0:
        print(f"WARNING: Impossible mutant score: {mutant_score}")
        return -1000  # Heavy penalty
    
    if wt_score < -15 or wt_score > 0:
        print(f"WARNING: Impossible WT score: {wt_score}") 
        return -1000
    
    # Rest of your scoring logic...
    specificity = wt_score - mutant_score
    
    # Moderate affinity target (-8 to -6 kcal/mol ≈ 1-100 µM)
    if -8.0 < mutant_score < -6.0:
        affinity_score = 1.0
    else:
        affinity_score = 0.0
    
    return 0.7 * specificity + 0.3 * affinity_score


def simple_temporary_score(mutant_score, wt_score):
    """
    Simplified scoring using only basic docking results
    """
    # 1. High specificity to mutant
    specificity = wt_score - mutant_score
    
    # 2. Moderate affinity (target -7 to -9 kcal/mol ≈ 1-100 µM)
    if -9.0 < mutant_score < -7.0:
        affinity_penalty = 0.0  # Perfect range
    else:
        affinity_penalty = abs(mutant_score + 8.0)  # Penalize being too strong/weak
    
    total_score = specificity - (0.3 * affinity_penalty)
    return total_score