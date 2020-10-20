def walk_sim(N_sim, N, mu, T, sigma, X_init):
    # N_sim - number of simulations
    # N - number of timesteps
    # mu - drift
    # T - expiry time
    # sigma - volatility
    # X_init - initial value
    import numpy as np
    import math
    delt = T/N
    up = sigma*math.sqrt(delt)
    down = -sigma*math.sqrt(delt)
    p = 1./2.*(1. + mu/sigma*math.sqrt(delt))
    X_new = np.ones(N_sim)*X_init
    ptest = np.zeros(N_sim)
    for i in np.arange(N): #timestep loop
        # now for each timestep, generate info
        # for all simulations
        ptest = np.random.uniform(0,1,N_sim)
        ptest = (ptest <= p)
        X_new = X_new + ptest*up + (1-ptest)*down
        # note: asterisks here perform *elementwise*
        # multiplication on numpy vectors, not
        # a dot product.
        # end of generation of data for all simulations
        # for this timestep
    return X_new
