# Compute Black-Scholes option value using a binomial tree - European case
# vectorized code

import numpy as np

S0 = 100 # S0 - current stock price
K = 100; # K - strike
T = 0.75; # T - expiry time
r = .03; # r - interest rate
sigma = .35; # sigma - volatility
opttype = 1; # opttype - 0 for a call, otherwise a put
Nsteps = 500 # Nsteps - number of timesteps

delt = T/Nsteps;

# tree parameters
u = np.exp(sigma * np.sqrt(delt) );
d = 1./u;
a = np.exp( r*delt );
p = (a - d)/(u - d);

# payoff at t=T
W = S0 * d**(np.arange(Nsteps,-1,-1)) * u**(np.arange(Nsteps+1))

# W is column vector of length Nsteps+1
if opttype == 0:
    W = np.maximum( W - K, 0);
else:W = np.maximum( K - W, 0);
    
# backward recursion
for i in np.arange(Nsteps,0,-1):
    W = np.exp(-r*delt)*( p*W[1:i+1] + (1-p)*W[0:i] )
    
print("Tree value: " + str(W[0]))
