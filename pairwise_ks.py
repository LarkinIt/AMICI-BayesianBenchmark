import glob
import gzip
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from modelproblem import ModelProblem
from result_classes import Result,MethodResults
from tqdm import tqdm

prob_name = "EGFR"
methods = [ "ptmcmc", "smc", "pmc"]

mod_prob = ModelProblem(prob_name)
mod_prob.initialize()

grouped_results = [MethodResults(x) for x in methods]

for method, group_obj in zip(methods, grouped_results):
	result_dir = f"results/{prob_name}/{method}/"
	fnames = glob.glob(result_dir + "*.pkl")
	for fname in fnames:
		#print(fname)
		with gzip.open(fname, "rb") as f:
			results = pickle.load(f)
		result_obj = Result(results)
		if result_obj.converged:
			group_obj.add_result(result_obj)
print(type(mod_prob.problem))			

fixed_idxs = mod_prob.problem.x_fixed_indices
par_names = mod_prob.problem.x_names
x=np.array(par_names)
mask=np.full(len(par_names),True,dtype=bool)
mask[fixed_idxs]=False
fit_par_names=x[mask]

all_ks_stats=[]

print(fit_par_names)
ks_df = pd.DataFrame(columns=["Param", "Method", "KS"])
for i, name in tqdm(enumerate(fit_par_names), desc="Parameter"):
	# assumes all runs have the same model parameters
	for runs in tqdm(grouped_results, desc="\tMethod"):
		ks_stats , pvals = runs.calc_pairwise_matrix(par_index=i)
		all_ks_stats.append(ks_stats[np.triu_indices(ks_stats.shape[0], k = 1)])
		for run_ks in ks_stats:
			for ks_stat in run_ks:
				new_row = {"Param":name, "Method":runs.abbr, "KS":ks_stat}
				ks_df.loc[len(ks_df)] = new_row

ks_df = ks_df.drop(ks_df[ks_df["KS"] == 0].index)
ks_df.to_csv(f"results/{prob_name}_ks_data.csv")
sns.violinplot(data=ks_df, x="Param", y="KS", hue="Method")
plt.savefig("ks_test.png")