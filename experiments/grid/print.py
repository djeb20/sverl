import pickle
import numpy as np

# Define mappings
state_map = {
    (0, 0): 1,
    (1, 0): 2
}

# Load dictionaries
with open("PolicyCharacteristic.pkl", "rb") as f:
    policy_char = pickle.load(f)

with open("PerformanceCharacteristic.pkl", "rb") as f:
    perf_char = pickle.load(f)

with open("PerformanceShapley.pkl", "rb") as f:
    shap = pickle.load(f)

# --------------------------- Policy Characteristic ---------------------------

# Print header
header = f"{'State':<6} | {'Action':<6} | {'Both':<5} | {'x':<5} | {'y':<5} | {'∅':<5}"
print("\n" + header)
print("-" * len(header))

# Print rows
for s, state in state_map.items():
    for a_idx, a in enumerate(['N', 'E', 'S', 'W']):
        
        chars = [policy_char[*np.where([True, True], s, -100)][a_idx],
                 policy_char[*np.where([True, False], s, -100)][a_idx],
                 policy_char[*np.where([False, True], s, -100)][a_idx],
                 policy_char[*np.where([False, False], s, -100)][a_idx]]
        
        print(f"{str(state):<6} | {a:<6} | "
              f"{chars[0]:<5} | {chars[1]:<5} | {chars[2]:<5.2f} | {chars[3]:<5.2f} ")
    print()

# --------------------------- Performance Characteristic + Shapley ---------------------------

# Print header
header = f"{'State':<6} | {'p^pi(s)':<7} | {'Both':<5} | {'d1':<5} | {'d2':<5} | {'∅':<5} | {'phi_d1':<7} | {'phi_d2':<7}"
print("\n" + header)
print("-" * len(header))

# Print rows
for (s, state), steady_state in zip(state_map.items(), ["1/7", "2/7"]):
                
    chars = [perf_char[True, True][s][0],
             perf_char[True, False][s][0],
             perf_char[False, True][s][0],
             perf_char[False, False][s][0]]
        
    print(f"{str(state):<6} | {steady_state:<7} | "
            f"{chars[0]:<5.2f} | {chars[1]:<5.2f} | {chars[2]:<5.2f} | {chars[3]:<5.2f} | "
            f"{shap[s][0][0]:<7.2f} | {shap[s][1][0]:<7.2f}")
    
    print()
