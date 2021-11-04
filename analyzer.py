import pandas as pd

update1 = pd.read_csv('saved_logs/results_n=10000_K=2500_update=1_alpha=1.0e-5_step=4.csv', delimiter=',')
# update2 = pd.read_csv('results/pd_Q/n=50/logs/results_n=50_K=40_update=1_defl=true.csv', delimiter=',')
# update3 = pd.read_csv('results/pd_Q/n=50/logs/results_n=50_K=40_update=2_defl=false.csv', delimiter=',')

print("Total iterations\n")

print(f"Update 1: {update1.shape}\n")
# print(f"Update 2: {update2.shape}\n")
# print(f"Update 3: {update3.shape}\n")


print("Total time\n")

print(f"Update 1: {update1['Time'].sum()}\n")
# print(f"Update 3: {update3['Time'].sum()}\n")

print("\n\n")

print("Best dual gap and dual value:\n")

min_index = update1.loc[update1['Dual_gap'] > 0, 'Dual_gap'].idxmin()

print(f"Update 1, best dual gap: {update1.loc[min_index, 'Dual_gap']}\n")
print(f"Update 1, best dual value: {update1.loc[min_index, 'DualValue']}\n")
print(f"Update 1, best iteration: {update1.loc[min_index, 'Iteration']}\n")
print(f"Update 1, best \lambda-norm: {update1.loc[min_index, 'λ_norm_residual']}\n")
print(f"Update 1, best x-norm: {update1.loc[min_index, 'x_norm_residual']}\n")

# min_index = update3[['Dual_gap']].idxmin()

# print(f"Update 1 deflection false, best dual gap: {update3.loc[min_index, 'Dual_gap']}\n")
# print(f"Update 1 deflection false, best dual value: {update3.loc[min_index, 'DualValue']}\n")
