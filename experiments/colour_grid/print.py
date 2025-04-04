import pickle
import numpy as np

# Define mappings
state_map = {
    (1, 1): (1, 'Red'),
    (3, 0): (3, 'Green')
}

# Load dictionaries
with open("PolicyCharacteristic.pkl", "rb") as f:
    char = pickle.load(f)

with open("PolicyShapley.pkl", "rb") as f:
    shap = pickle.load(f)

# Print header
header = f"{'State':<12} | {'Action':<6} | {'p^pi(s)':<7} | {'Both':<5} | {'I':<5} | {'C':<5} | {'∅':<5} | {'phi_I':<7} | {'phi_C':<7}"
print("\n" + header)
print("-" * len(header))

# Print rows
for s, state in state_map.items():
    for a_idx, a in enumerate(['N', 'E', 'S', 'W']):
        
        chars = [char[*np.where([True, True], s, -100)][a_idx],
                 char[*np.where([True, False], s, -100)][a_idx],
                 char[*np.where([False, True], s, -100)][a_idx],
                 char[*np.where([False, False], s, -100)][a_idx]]
        
        print(f"{str(state):<12}| {a:<6} | {0.25:<7.2f} | "
              f"{chars[0]:<5} | {chars[1]:<5} | {chars[2]:<5} | {chars[3]:<5} | "
              f"{shap[s][0][a_idx]:<7.3f} | {shap[s][1][a_idx]:<7.3f}")
    print()
