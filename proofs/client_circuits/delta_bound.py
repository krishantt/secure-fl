from pysnark.runtime import PrivVal, PubVal, snark
from pysnark.branching import if_then_else

@snark
def delta_bound_proof(initial, updated, bound):
    """
    PySNARK circuit for delta bound verification.
    
    Args:
        initial: List of PrivVal (already wrapped)
        updated: List of PrivVal (already wrapped)  
        bound: PubVal (already wrapped)
    
    Returns:
        l2_sq: The computed L2 norm squared
    """
    n = len(initial)
    deltas = [updated[i] - initial[i] for i in range(n)]
    
    # Compute squared L2 norm
    l2_sq = sum([d * d for d in deltas])
    
    # Compute bound squared
    bound_sq = bound * bound
    
    # Create constraint: if l2_sq <= bound_sq, output 1, else 0
    valid = if_then_else(l2_sq <= bound_sq, PrivVal(1), PrivVal(0))
    
    # Make valid equal to 1 (this creates the constraint)
    valid.assert_eq(PrivVal(1))
    
    return l2_sq