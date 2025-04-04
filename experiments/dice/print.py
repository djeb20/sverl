import pickle
import numpy as np

# States to explain
states =[(3, 6), (1, 1)]

# Load dictionaries
with open("ValueCharacteristic.pkl", "rb") as f:
    char = pickle.load(f)

with open("ValueShapley.pkl", "rb") as f:
    shap = pickle.load(f)

# Print header
header = f"{'State':<12} | {'p^pi(s)':<7} | {'Both':<5} | {'d1':<5} | {'d2':<5} | {'∅':<5} | {'phi_d1':<7} | {'phi_d2':<7}"
print("\n" + header)
print("-" * len(header))

# Print rows
for s in states:
                
    chars = [char[*np.where([True, True], s, -100)][0],
             char[*np.where([True, False], s, -100)][0],
             char[*np.where([False, True], s, -100)][0],
             char[*np.where([False, False], s, -100)][0]]
        
    print(f"{str(s):<12} | {"1/36":<7} | "
            f"{chars[0]:<5.2f} | {chars[1]:<5.2f} | {chars[2]:<5.2f} | {chars[3]:<5.2f} | "
            f"{shap[s][0][0]:<7.2f} | {shap[s][1][0]:<7.2f}")
    
    print()
