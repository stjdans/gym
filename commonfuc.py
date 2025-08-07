import numpy as np

def rrand(list):
    if np.max(list) == 0: 
        return np.random.choice(range(4))
    else:
        return np.argmax(list)
    
    
def one_hot(n, x):
    return np.identity(n)[x:x+1]