# Write a Python function that computes the conditional probability P(A|B), 
# given a joint probability distribution over events A and B. 
# The distribution is provided as a dictionary with keys 
# ('A','B'), ('A','`B'), ('`A','B'), ('`A','`B'), 
# where the backtick ` denotes logical NOT.

def conditional_probability(joint_distribution: dict) -> float:
    """
    Compute conditional probability P(A|B) from a joint probability distribution.

    Args:
        joint_distribution (dict): dictionary with keys ('A','B'), ('A','`B'), ('`A','B'), ('`A','`B')

    Returns:
        float: Conditional probability P(A|B)
    """
    # Your code here
    # P(A|B) = P(B|A) * P(A) / P(B)
    
    # P(A) = P(A and B) + P(A and not B)
    P_A = joint_distribution[('A', 'B')] + joint_distribution[('A', '`B')]

    # P(B) = P(A and B) + P(not A and B)
    P_B = joint_distribution[('A', 'B')] + joint_distribution[('`A', 'B')]

    # P(B|A) = P(A and B) / P(A)
    P_B_given_A = joint_distribution[('A', 'B')] / P_A if P_A != 0 else 0

    # P(A|B) = P(B|A) * P(A) / P(B)
    P_A_given_B = (P_B_given_A * P_A) / P_B if P_B != 0 else 0

    return P_A_given_B


# test case
res = conditional_probability({('A','B'):0.2, ('A','`B'):0.3, ('`A','B'):0.1, ('`A','`B'):0.4})
print(res)  # Expected output: 0.6666666666666666