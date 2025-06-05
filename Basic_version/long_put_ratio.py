import numpy as np
import matplotlib.pyplot as plt

# Input parameters
q_long = float(input("Enter quantity of long puts: "))
K_long = float(input("Enter strike price of long puts: "))
P_long = float(input("Enter price of long puts: "))
q_short = float(input("Enter quantity of short puts: "))
K_short = float(input("Enter strike price of short puts: "))
P_short = float(input("Enter price of short puts: "))

# Calculate net premium
net_premium = q_short * P_short - q_long * P_long

# Generate price range
S = np.linspace(min(K_long, K_short) - 0.02, max(K_long, K_short) + 0.02, 1000)

# Calculate payoff
payoff = (q_long * np.maximum(K_long - S, 0) - 
          q_short * np.maximum(K_short - S, 0) + 
          net_premium)

# Find breakeven points
sign_changes = np.where(np.diff(np.sign(payoff)))[0]
breakevens = []
for idx in sign_changes:
    x1, x2 = S[idx], S[idx+1]
    y1, y2 = payoff[idx], payoff[idx+1]
    if y1 != y2:
        breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
        breakevens.append(breakeven)

# Calculate max profit
max_profit = np.max(payoff)
max_profit_idx = np.argmax(payoff)
max_profit_S = S[max_profit_idx]

# Plot setup
plt.figure(figsize=(12, 7))
plt.plot(S, payoff, color='black', linewidth=2, label='Payoff')
plt.axhline(0, color='black', linewidth=0.5)

# Color filling
plt.fill_between(S, payoff, where=payoff >= 0, color='green', alpha=0.3, label='Profit')
plt.fill_between(S, payoff, where=payoff < 0, color='red', alpha=0.3, label='Loss')

# Breakeven markers
for be in breakevens:
    plt.axvline(be, color='blue', linestyle='--', alpha=0.7)
    plt.text(be, 0, f' Breakeven\n{be:.4f}', verticalalignment='bottom', 
             horizontalalignment='center', color='blue')

# Max profit annotation
plt.annotate(f'Max Profit: {max_profit:.4f}',
             xy=(max_profit_S, max_profit),
             xytext=(max_profit_S, max_profit + 0.001),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

# Strike lines
plt.axvline(K_long, color='grey', linestyle='--', alpha=0.5)
plt.text(K_long, np.min(payoff), f' Long Put Strike\n{K_long:.4f}', 
         horizontalalignment='center', verticalalignment='top')
plt.axvline(K_short, color='grey', linestyle='--', alpha=0.5)
plt.text(K_short, np.min(payoff), f' Short Put Strike\n{K_short:.4f}', 
         horizontalalignment='center', verticalalignment='top')

# Formatting
plt.title('Long Put Ratio Spread Payoff Diagram')
plt.xlabel('Underlying Asset Price')
plt.ylabel('Profit/Loss')
plt.grid(True, alpha=0.3)
plt.legend()

# Format axes for small values
plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
plt.tight_layout()

plt.show()