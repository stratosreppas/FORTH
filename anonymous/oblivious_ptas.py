import numpy as np
from anonymous import anonymous_game

def moment_search(G, epsilon):
    # Step 1
    C = np.random.randint()
    k = np.floor(C/epsilon)

    # Step 2
    t = np.random.randint(0, G.n, size=1)
    t0 = np.random.randint(0, G.n-t, size=1)
    t1 = G.n-t-t0

    # Step 3a
    if( t > k^3):
        return None #########
    
    # Step 3b
    else:
        ts = np.random(t)
        tb = t-ts



