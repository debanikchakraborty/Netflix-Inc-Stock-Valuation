# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 21:00:55 2024

@author: deban
"""
import yfinance as yf
import numpy as np
import numba
import matplotlib.pyplot as plt

# Function to calculate price path, return final price of path
@numba.njit # Numba decorator to speed up the loop
def PricePath(spot, rf, vol, time, time_steps):
    dt = time / time_steps
    avg_prc = 0 #create a variable to calcualate average price
    price_path = [] #creating a list to store the spot_prices 

    for t in range(time_steps): #implementing the Brownian Motion
        Z = np.random.normal()
        spot *= np.exp((rf - (vol/2)) * dt + np.sqrt(vol) * np.sqrt(dt) * Z)
        avg_prc += spot
        price_path.append(spot)

    avg_prc /= time_steps
    return avg_prc, np.array(price_path)

# Parameters
spot_price = yf.Ticker('NFLX').history(period='1d')['Close'].iloc[-1] #retriving the current price
rf_rate = 0.04 #risk free rate in 2023 March
volatility = 0.22 #volatility of netflix stocks in 2023 March
simulations= 1000 #no of simulations
total_time = 1.0 # 1 year time
time_steps = 365 #daily price calcualting

# Generate multiple price paths by simulations
all_paths = []
for _ in range(simulations):
    _, path = PricePath(spot_price, rf_rate, volatility, total_time, time_steps)
    all_paths.append(path)


#Plot simulated daily prices on 365 days range
plt.figure(figsize=(10, 6))
for path in all_paths:
    plt.plot(range(0, time_steps), path, label='Price Path')

plt.title('Price Paths')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()