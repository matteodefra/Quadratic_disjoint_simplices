import pandas as pd

# n = 50, K = 40
# Positive definite Q   

update1_deffalse = pd.read_csv('results/pd_Q/n=50/logs/results_n=50_K=40_update=1_defl=false.csv', delimiter=',')
update1_deftrue = pd.read_csv('results/pd_Q/n=50/logs/results_n=50_K=40_update=1_defl=true.csv', delimiter=',')
update2_deffalse = pd.read_csv('results/pd_Q/n=50/logs/results_n=50_K=40_update=2_defl=false.csv', delimiter=',')
update2_deftrue = pd.read_csv('results/pd_Q/n=50/logs/results_n=50_K=40_update=2_defl=true.csv', delimiter=',')
update3_deffalse = pd.read_csv('results/pd_Q/n=50/logs/results_n=50_K=40_update=3_defl=false.csv', delimiter=',')
update3_deftrue = pd.read_csv('results/pd_Q/n=50/logs/results_n=50_K=40_update=3_defl=true.csv', delimiter=',')

# Positive semidefinite Q

psd_update1_deffalse = pd.read_csv('results/psd_Q/n=50/logs/results_n=50_K=40_update=1_defl=false.csv', delimiter=',')
psd_update1_deftrue = pd.read_csv('results/psd_Q/n=50/logs/results_n=50_K=40_update=1_defl=true.csv', delimiter=',')
psd_update2_deffalse = pd.read_csv('results/psd_Q/n=50/logs/results_n=50_K=40_update=2_defl=false.csv', delimiter=',')
psd_update2_deftrue = pd.read_csv('results/psd_Q/n=50/logs/results_n=50_K=40_update=2_defl=true.csv', delimiter=',')
psd_update3_deffalse = pd.read_csv('results/psd_Q/n=50/logs/results_n=50_K=40_update=3_defl=false.csv', delimiter=',')
psd_update3_deftrue = pd.read_csv('results/psd_Q/n=50/logs/results_n=50_K=40_update=3_defl=true.csv', delimiter=',')


# n = 100, K = 20 

# Positive definite Q   

# update1_deffalse = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=20_update=1_defl=false.csv', delimiter=',')
# update1_deftrue = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=20_update=1_defl=true.csv', delimiter=',')
# update2_deffalse = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=20_update=2_defl=false.csv', delimiter=',')
# update2_deftrue = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=20_update=2_defl=true.csv', delimiter=',')
# update3_deffalse = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=20_update=3_defl=false.csv', delimiter=',')
# update3_deftrue = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=20_update=3_defl=true.csv', delimiter=',')

# n = 100, K = 33

# update1_deffalse = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=33_update=1_defl=false.csv', delimiter=',')
# update1_deftrue = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=33_update=1_defl=true.csv', delimiter=',')
# update2_deffalse = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=33_update=2_defl=false.csv', delimiter=',')
# update2_deftrue = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=33_update=2_defl=true.csv', delimiter=',')
# update3_deffalse = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=33_update=3_defl=false.csv', delimiter=',')
# update3_deftrue = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=33_update=3_defl=true.csv', delimiter=',')

# n = 100, K = 50

# update1_deffalse = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=50_update=1_defl=false.csv', delimiter=',')
# update1_deftrue = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=50_update=1_defl=true.csv', delimiter=',')
# update2_deffalse = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=50_update=2_defl=false.csv', delimiter=',')
# update2_deftrue = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=50_update=2_defl=true.csv', delimiter=',')
# update3_deffalse = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=50_update=3_defl=false.csv', delimiter=',')
# update3_deftrue = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=50_update=3_defl=true.csv', delimiter=',')

# n = 100, K = 66

# update1_deffalse = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=66_update=1_defl=false.csv', delimiter=',')
# update1_deftrue = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=66_update=1_defl=true.csv', delimiter=',')
# update2_deffalse = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=66_update=2_defl=false.csv', delimiter=',')
# update2_deftrue = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=66_update=2_defl=true.csv', delimiter=',')
# update3_deffalse = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=66_update=3_defl=false.csv', delimiter=',')
# update3_deftrue = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=66_update=3_defl=true.csv', delimiter=',')

# n = 100, K = 80

# update1_deffalse = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=80_update=1_defl=false.csv', delimiter=',')
# update1_deftrue = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=80_update=1_defl=true.csv', delimiter=',')
# update2_deffalse = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=80_update=2_defl=false.csv', delimiter=',')
# update2_deftrue = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=80_update=2_defl=true.csv', delimiter=',')
# update3_deffalse = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=80_update=3_defl=false.csv', delimiter=',')
# update3_deftrue = pd.read_csv('results/pd_Q/n=100/logs/results_n=100_K=80_update=3_defl=true.csv', delimiter=',')

# n = 25, K = 13

# psd_update1_deffalse = pd.read_csv('results/psd_Q/n=25/logs/results_n=25_K=13_update=1_defl=false.csv', delimiter=',')
# psd_update1_deftrue = pd.read_csv('results/psd_Q/n=25/logs/results_n=25_K=13_update=1_defl=true.csv', delimiter=',')
# psd_update2_deffalse = pd.read_csv('results/psd_Q/n=25/logs/results_n=25_K=13_update=2_defl=false.csv', delimiter=',')
# psd_update2_deftrue = pd.read_csv('results/psd_Q/n=25/logs/results_n=25_K=13_update=2_defl=true.csv', delimiter=',')
# psd_update3_deffalse = pd.read_csv('results/psd_Q/n=25/logs/results_n=25_K=13_update=3_defl=false.csv', delimiter=',')
# psd_update3_deftrue = pd.read_csv('results/psd_Q/n=25/logs/results_n=25_K=13_update=3_defl=true.csv', delimiter=',')



print("Positive definite Q\n")

print("Total iterations\n")

print(f"Update 1 deflection false: {update1_deffalse.shape}\n")
print(f"Update 1 deflection true: {update1_deftrue.shape}\n")
print(f"Update 2 deflection false: {update2_deffalse.shape}\n")
print(f"Update 2 deflection true: {update2_deftrue.shape}\n")
print(f"Update 3 deflection false: {update3_deffalse.shape}\n")
print(f"Update 3 deflection true: {update3_deftrue.shape}\n")


print("Total time\n")

print(f"Update 1 deflection false: {update1_deffalse['Time'].sum()}\n")
print(f"Update 1 deflection true: {update1_deftrue['Time'].sum()}\n")
print(f"Update 2 deflection false: {update2_deffalse['Time'].sum()}\n")
print(f"Update 2 deflection true: {update2_deftrue['Time'].sum()}\n")
print(f"Update 3 deflection false: {update3_deffalse['Time'].sum()}\n")
print(f"Update 3 deflection true: {update3_deftrue['Time'].sum()}\n")

print("\n\n")

print("Best dual gap and dual value:\n")

min_index = update1_deffalse[['Dual_gap']].idxmin()

print(f"Update 1 deflection false, best dual gap: {update1_deffalse.loc[min_index, 'Dual_gap']}\n")
print(f"Update 1 deflection false, best dual value: {update1_deffalse.loc[min_index, 'DualValue']}\n")

min_index = update1_deftrue[['Dual_gap']].idxmin()

print(f"Update 1 deflection true, best dual gap: {update1_deftrue.loc[min_index, 'Dual_gap']}\n")
print(f"Update 1 deflection true, best dual value: {update1_deftrue.loc[min_index, 'DualValue']}\n")

min_index = update2_deffalse[['Dual_gap']].idxmin()

print(f"Update 2 deflection false, best dual gap: {update2_deffalse.loc[min_index, 'Dual_gap']}\n")
print(f"Update 2 deflection false, best dual value: {update2_deffalse.loc[min_index, 'DualValue']}\n")

min_index = update2_deftrue[['Dual_gap']].idxmin()

print(f"Update 2 deflection true, best dual gap: {update2_deftrue.loc[min_index, 'Dual_gap']}\n")
print(f"Update 2 deflection true, best dual value: {update2_deftrue.loc[min_index, 'DualValue']}\n")

min_index = update3_deffalse[['Dual_gap']].idxmin()

print(f"Update 3 deflection false, best dual gap: {update3_deffalse.loc[min_index, 'Dual_gap']}\n")
print(f"Update 3 deflection false, best dual value: {update3_deffalse.loc[min_index, 'DualValue']}\n")

min_index = update3_deftrue[['Dual_gap']].idxmin()

print(f"Update 3 deflection true, best dual gap: {update3_deftrue.loc[min_index, 'Dual_gap']}\n")
print(f"Update 3 deflection true, best dual value: {update3_deftrue.loc[min_index, 'DualValue']}\n")

print("\n\n")

print("Positive semidefinite Q\n")

print("Total iterations\n")

print(f"Update 1 deflection false: {psd_update1_deffalse.shape}\n")
print(f"Update 1 deflection true: {psd_update1_deftrue.shape}\n")
print(f"Update 2 deflection false: {psd_update2_deffalse.shape}\n")
print(f"Update 2 deflection true: {psd_update2_deftrue.shape}\n")
print(f"Update 3 deflection false: {psd_update3_deffalse.shape}\n")
print(f"Update 3 deflection true: {psd_update3_deftrue.shape}\n")


print("Total time\n")

print(f"Update 1 deflection false: {psd_update1_deffalse['Time'].sum()}\n")
print(f"Update 1 deflection true: {psd_update1_deftrue['Time'].sum()}\n")
print(f"Update 2 deflection false: {psd_update2_deffalse['Time'].sum()}\n")
print(f"Update 2 deflection true: {psd_update2_deftrue['Time'].sum()}\n")
print(f"Update 3 deflection false: {psd_update3_deffalse['Time'].sum()}\n")
print(f"Update 3 deflection true: {psd_update3_deftrue['Time'].sum()}\n")

print("\n\n")

print("Best dual gap and dual value:\n")

min_index = psd_update1_deffalse[['Dual_gap']].idxmin()

print(f"Update 1 deflection false, best dual gap: {psd_update1_deffalse.loc[min_index, 'Dual_gap']}\n")
print(f"Update 1 deflection false, best dual value: {psd_update1_deffalse.loc[min_index, 'DualValue']}\n")

min_index = psd_update1_deftrue[['Dual_gap']].idxmin()

print(f"Update 1 deflection true, best dual gap: {psd_update1_deftrue.loc[min_index, 'Dual_gap']}\n")
print(f"Update 1 deflection true, best dual value: {psd_update1_deftrue.loc[min_index, 'DualValue']}\n")

min_index = psd_update2_deffalse[['Dual_gap']].idxmin()

print(f"Update 2 deflection false, best dual gap: {psd_update2_deffalse.loc[min_index, 'Dual_gap']}\n")
print(f"Update 2 deflection false, best dual value: {psd_update2_deffalse.loc[min_index, 'DualValue']}\n")

min_index = psd_update2_deftrue[['Dual_gap']].idxmin()

print(f"Update 2 deflection true, best dual gap: {psd_update2_deftrue.loc[min_index, 'Dual_gap']}\n")
print(f"Update 2 deflection true, best dual value: {psd_update2_deftrue.loc[min_index, 'DualValue']}\n")

min_index = psd_update3_deffalse[['Dual_gap']].idxmin()

print(f"Update 3 deflection false, best dual gap: {psd_update3_deffalse.loc[min_index, 'Dual_gap']}\n")
print(f"Update 3 deflection false, best dual value: {psd_update3_deffalse.loc[min_index, 'DualValue']}\n")

min_index = psd_update3_deftrue[['Dual_gap']].idxmin()

print(f"Update 3 deflection true, best dual gap: {psd_update3_deftrue.loc[min_index, 'Dual_gap']}\n")
print(f"Update 3 deflection true, best dual value: {psd_update3_deftrue.loc[min_index, 'DualValue']}\n")

print("\n\n")