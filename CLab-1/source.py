import numpy as np

########################################################
## Complete functions in skeleton codes below
## following instructions in each function.
## Do not modify existing function name or inputs.
## Do not test your codes here - use main.py instead.
## You may use any built-in functions from NumPy.
## You may define and call new functions as you see fit.
########################################################


def low_rank_approx(A, k):
    '''
    inputs: 
      - A: m-by-n matrix
      - k: positive integer, k<=m, k<=n
    returns:
      X: m-by-n matrix that is an as-close-as-possible approximation of A
         up to rank k
    '''
    u, s, vt = np.linalg.svd(A)
    u_new = u[:,:k]
    s_new = np.diag(s[:k])
    vt_new = vt[:k]

    X =u_new @ vt_new
    return X
    
def constrained_LLS(A, B):
    '''
    inputs:
      - A: n-by-n full rank matrix
      - B: n-by-n matrix
    returns:
      x: n-diemsional vector that minimises ||Ax||2 subject to ||Bx||2=1 
    '''

    eps=0.00000001
    u, s, vt = np.linalg.svd(B)
    s+=eps
    s1=np.diag(s)
    temp = s1@vt
    new_matrix = A@temp.T # making sure the matrix is full rank? Didn't catch on

    u2, s2, vt2 = np.linalg.svd(new_matrix)

    x = vt.T@s1@vt2[-1]
    return x
### you can optionally write your own functions like below ###

# def my_func_name(input1, input2, ...):
#     do something
#     return ...