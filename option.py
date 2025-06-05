import numpy as np
import matplotlib.pyplot as plt

# Collect strategy details from user
options = []
num_options = int(input("Enter the number of options in the strategy: "))

for i in range(num_options):
    print(f"\nOption {i+1}:")
    quantity = float(input("Enter quantity: "))
    action = input("Enter 'buy' or 'sell': ").lower()
    option_type = input("Enter 'call' or 'put': ").lower()
    strike = float(input("Enter strike price: "))
    premium = float(input("Enter premium paid/received per option: "))2

    options.append({
        'quantity': quantity,
        'action': action,
        'type': option_type,
        'strike': strike,
        'premium': premium
    })

# Generate spot price range
strikes = [option['strike'] for option in options]
min_strike = min(strikes)
max_strike = max(strikes)
spot_range = np.linspace(min_strike - 20, max_strike + 20, 1000)

total_payoff = np.zeros(len(spot_range))

# Calculate payoff for each option and accumulate
for option in options:
    quantity = option['quantity']
    action = option['action']
    option_type = option['type']
    strike = option['strike']
    premium = option['premium']
    
    # Calculate intrinsic value
    if option_type == 'call':
        intrinsic = np.maximum(spot_range - strike, 0)
    else:
        intrinsic = np.maximum(strike - spot_range, 0)
    
    # Calculate position payoff
    if action == 'buy':
        payoff = intrinsic - premium
    else:
        payoff = premium - intrinsic
    
    # Add to total payoff
    total_payoff += quantity * payoff

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(spot_range, total_payoff, label='Total Profit/Loss', color='b')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Spot Price at Expiration')
plt.ylabel('Total Profit/Loss')
plt.title('Options Strategy Payoff Diagram')
plt.legend()
plt.grid(True)
plt.show()