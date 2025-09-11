import json
import matplotlib.pyplot as plt
import numpy as np

# Load the data
with open('/Users/ahmed.elgazzar/Code/flow_time/nfe_efficiency/lorenz/NFE Efficiency Analysis - Lorenz_sweep_results.json', 'r') as f:
    data = json.load(f)

# Extract NFE values and CRPS prediction scores
nfe_values = []
afm_crps = []
fm_crps = []

for key, value in data.items():
    if 'nfe_' in key and 'timing' in value:
        nfe = value['timing']['nfe']
        nfe_values.append(nfe)
        afm_crps.append(value['AFM']['crps_prediction'])
        fm_crps.append(value['FM']['crps_prediction'])

# Sort by NFE values
sorted_indices = np.argsort(nfe_values)
nfe_values = np.array(nfe_values)[sorted_indices]
afm_crps = np.array(afm_crps)[sorted_indices]
fm_crps = np.array(fm_crps)[sorted_indices]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(nfe_values, afm_crps, 'o-', label='AFM', linewidth=2, markersize=8)
plt.plot(nfe_values, fm_crps, 's-', label='FM', linewidth=2, markersize=8)

plt.xlabel('Number of Function Evaluations (NFE)', fontsize=12)
plt.ylabel('Test CRPS', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('/Users/ahmed.elgazzar/Code/flow_time/nfe_vs_crps_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"NFE values: {nfe_values}")
print(f"AFM CRPS: {afm_crps}")
print(f"FM CRPS: {fm_crps}")