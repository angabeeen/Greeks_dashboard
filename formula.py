import math
from scipy.stats import norm
import numpy as np

import matplotlib.pyplot as plt
import numpy as np




def black_scholes(K,S,r,T,sigma, type="call"):
    """
    S = spot price
    K = strike price  
    T = time to expiry (in years)
    r = risk free rate
    sigma = implied volatility
    """
    d1= (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

    d2 = d1 - sigma*np.sqrt(T)
    if type=="call":
        c=S*norm.cdf(d1)-(K*np.exp(-r*T)*norm.cdf(d2))
       
    else:
        c = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


    


    return c

def greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Delta — sensitivity to spot price
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Gamma — rate of change of delta (same for call and put)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta — time decay (per day)
    theta_call = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                  - r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
    
    # Vega — sensitivity to volatility (per 1% move in vol)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    # Rho — sensitivity to interest rates (per 1% move)
    if option_type == "call":
        rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100
    
    return {"delta": delta, "gamma": gamma, 
            "theta": theta_call, "vega": vega, "rho": rho}




spots = np.linspace(50, 150, 100)  # range of spot prices
K, T, r, sigma = 100, 1, 0.05, 0.2

deltas = [greeks(S, K, T, r, sigma)["delta"] for S in spots]
gammas = [greeks(S, K, T, r, sigma)["gamma"] for S in spots]
thetas= [greeks(S, K, T, r, sigma)["theta"] for S in spots]
vegas= [greeks(S, K, T, r, sigma)["vega"] for S in spots]
rhos= [greeks(S, K, T, r, sigma)["rho"] for S in spots]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Options Greeks vs Spot Price")

axes[0,0].plot(spots, deltas)
axes[0,0].set_title("Delta")
axes[0,0].axvline(x=100, linestyle='--', color='red', label='ATM')
axes[0,1].plot(spots, gammas)
axes[0,1].set_title("Gamma")
axes[0,1].axvline(x=100, linestyle='--', color='blue', label='ATM')

axes[0,2].plot(spots, thetas)
axes[0,2].set_title("Theta")
axes[0,2].axvline(x=100, linestyle='--', color='green', label='ATM')

axes[1,0].plot(spots, vegas)
axes[1,0].set_title("Vegas")
axes[1,0].axvline(x=100, linestyle='--', color='orange', label='ATM')

axes[1,1].plot(spots, rhos)
axes[1,1].set_title("Rho")
axes[1,1].axvline(x=100, linestyle='--', color='black', label='ATM')

plt.show()