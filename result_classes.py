import itertools
import numpy as np
from scipy.stats import distributions

LLH_CONVERGENCE_THRESHOLD = {
	"Zhao_QuantBiol2020": -700,
	"EGFR": 85,
	"Raia_CancerResearch2011": -480
}

class Result:
	def __init__(self, result_dict, use_old_burnin=False) -> None:
		for key in result_dict:
			setattr(self, key, result_dict[key])
		if self.method != "ptmcmc":
			self.converged = True
		
		else:
			if use_old_burnin:
				if self.converged:
					burn_in_idx = self.algo_specific_info["burn_in_idx"]
					n_chains = self.n_chains
					self.n_fun_calls = (burn_in_idx+1)*n_chains
			else:
				threshold = LLH_CONVERGENCE_THRESHOLD[self.problem]
				#print(f"For the {self.problem} problem, the llh threshold = {threshold}")
				burn_in_idx = self.check_ptmcmc_convergence(threshold)
				print(f"The burn-in index for run {self.seed} is {burn_in_idx} with threshold = {threshold}")
				self.algo_specific_info["burn_in_idx"] = int(burn_in_idx)
				if burn_in_idx == -1:
					self.converged = False
				else:
					self.create_ptmcmc_posterior_ensemble()

		if not("posterior_weights" in result_dict.keys()):
			n = len(result_dict["posterior_llhs"])
			self.posterior_weights = np.array([1.0/n]*n)


	# Courtesy of ChatGPT
	def largest_step_size(self, L, N):
		# Start from the largest possible step size (L-1) and work backwards
		for i in range((L - 1) // (N - 1), 0, -1):
			# Check if this step size allows exactly N samples within the bounds of L
			if (N - 1) * i < L:
				return i
		return None  # Return None if no valid step size is found


	def create_ptmcmc_posterior_ensemble(self):
		# get the lowest temperature chain index
		# Note: remember that beta is the inverse of the temperature so 
		# we want the chain with the max beta
		ch_idx = np.argmax(self.algo_specific_info["betas"])
		diff = (self.n_iter - self.algo_specific_info["burn_in_idx"]) // self.sample_step

		scaled_burn_in_idx = int(self.algo_specific_info["burn_in_idx"] // self.sample_step)
		print(scaled_burn_in_idx, self.n_iter, diff, self.n_ensemble)
		if diff < self.n_ensemble:
			self.converged = False
			# make empty lists to avert downstream errors
			self.posterior_samples = np.empty((self.n_ensemble, self.all_samples.shape[-1]))
			self.posterior_llhs = np.empty(self.n_ensemble)
			self.posterior_priors = np.empty(self.n_ensemble)
		else:
			step_size = self.largest_step_size(diff, self.n_ensemble)
			if not step_size:
				self.converged = False
			else:
				print(self.all_samples.shape, self.all_llhs.shape, scaled_burn_in_idx)
				trim_trace_x = self.all_samples[scaled_burn_in_idx:, ch_idx, :]
				trim_trace_llhs = -1*self.all_llhs[scaled_burn_in_idx:, ch_idx]
				trim_trace_priors = self.all_priors[scaled_burn_in_idx:, ch_idx]

				print(step_size, trim_trace_x.shape)

				# Overwrite posterior ensemble with rolling averages burn-in index
				self.posterior_samples = trim_trace_x[::step_size, :][:self.n_ensemble, :]
				self.posterior_llhs = trim_trace_llhs[::step_size][:self.n_ensemble]
				self.posterior_priors = trim_trace_priors[::step_size][:self.n_ensemble]

				print(self.posterior_samples.shape)
				# Calculate the number of function callsd
				n_chains = self.n_chains
				self.n_fun_calls = (self.algo_specific_info["burn_in_idx"]+1)*n_chains


	def check_ptmcmc_convergence(self, threshold, window_size=100):
		# get the lowest temperature chain 
		chain_llhs = self.all_llhs[:, 0]

		if len(chain_llhs) < window_size:
			return -1

		# Compute rolling averages
		rolling_avgs = np.convolve(chain_llhs, np.ones(window_size)/window_size, mode='valid')

		for i in range(len(rolling_avgs)):
			if rolling_avgs[i] > threshold:
				# From here onward, the avg should stay above threshold
				if all(avg > threshold for avg in rolling_avgs[i:]):
					# multiply the index by the downsampling step size
					downsample_step_size = self.sample_step
					idx = downsample_step_size * (i + (window_size-1))
					return idx
		return -1


	def get_sampling_ratio(self, par_bounds, par_idx=0, unlog=True) -> float:
		"""
		Measures the ratio of the sampling space 
		explored for a given parameter index
		"""
		bound_diff = par_bounds[par_idx][1] - par_bounds[par_idx][0]
		#print(self.all_samples.shape)
		par_samples = self.all_samples[:, :, par_idx]
		if unlog:
			par_samples = 10**par_samples
		#print("HERE: ", par_samples)
		#print(bound_diff)
		max_val = np.max(par_samples)
		min_val = np.min(par_samples)
		sample_diff = max_val - min_val
		#print(sample_diff)
		return sample_diff/bound_diff
	
	def get_convergence(self, llh_threshold):
		conv_calls = np.nan
		if self.converged:
			try:
				idxs = np.where(self.all_llhs > llh_threshold)
				first_iter = np.min(idxs[0])
			except ValueError:
				first_iter = self.n_iter-1

			if self.method == "ptmcmc":
				#print(first_iter)
				conv_calls = (first_iter+1) * self.n_chains
			else:
				conv_calls = self.algo_specific_info["calls_by_iter"][first_iter]
		return conv_calls

	def get_init_best_llh(self):
		all_llhs = self.all_llhs
		if self.method != "ptmcmc":
			iter0 = all_llhs[0,:]
		else:
			# this assumes 4 chains
			iter0 = all_llhs[:250,:]
		return np.amax(iter0)

	def get_max_llh(self):
		if self.converged:
			return max(self.posterior_llhs)
		else:
			ch_idx = np.argmax(self.algo_specific_info["betas"])
			#print(self.all_llhs.shape)
			# subsample chain
			n_samples = self.n_ensemble
			subsamples = np.random.choice(self.all_llhs[:, ch_idx],
								 			size=n_samples,
											replace=False)
			return np.amax(subsamples)

class MethodResults:
	def __init__(self, method) -> None:
		self.all_runs = []
		self.method = method
		if method == "pmc":
			self.abbr = "PMC"
			self.label = "Preconditioned Monte Carlo"
		elif method == "smc":
			self.abbr = "SMC"
			self.label = "Sequential Monte Carlo"
		elif method == "ptmcmc":
			self.abbr = "PT-MCMC"
			self.label = "Parallel Tempering MCMC"
	
	def add_result(self, result_obj):
		self.all_runs.append(result_obj)

	def get_fun_calls(self) -> np.array:
		all_calls = [x.n_fun_calls if x.converged else np.nan for x in self.all_runs]
		return np.array(all_calls)
	
	def get_llhs(self) -> np.array:
		all_llhs = [x.posterior_llhs if x.converged else [-1*np.inf]*x.n_ensemble for x in self.all_runs]
		return np.array(all_llhs)
	
	def get_avg_llhs(self) -> np.array:
		avgs = [np.average(x.posterior_llhs, weights=x.posterior_weights) if x.converged else -1*np.inf for x in self.all_runs]
		return np.array(avgs)

	def get_sampling_efficiency(self, bounds, par_idx) -> np.array:
		all_ratios = [x.get_sampling_ratio(bounds, par_idx) if x.converged else np.nan for x in self.all_runs]
		return np.array(all_ratios)
	
	def get_convergence_times(self, llh_threshold):
		all_convs = [x.get_convergence(llh_threshold) for x in self.all_runs]
		return np.array(all_convs)

	def get_best_inits(self):
		init_llhs = [x.get_init_best_llh() for x in self.all_runs]
		return np.array(init_llhs)

	def get_max_llhs(self):
		max_llhs = [x.get_max_llh() for x in self.all_runs]
		return np.array(max_llhs)

	# Source: https://stackoverflow.com/questions/40044375/how-to-calculate-the-kolmogorov-smirnov-statistic-between-two-weighted-samples
	def ks_weighted(self, data1, data2, wei1, wei2, alternative='two-sided'):
		ix1 = np.argsort(data1)
		ix2 = np.argsort(data2)
		data1 = data1[ix1]
		data2 = data2[ix2]
		wei1 = wei1[ix1]
		wei2 = wei2[ix2]
		data = np.concatenate([data1, data2])
		cwei1 = np.hstack([0, np.cumsum(wei1)/sum(wei1)])
		cwei2 = np.hstack([0, np.cumsum(wei2)/sum(wei2)])
		cdf1we = cwei1[np.searchsorted(data1, data, side='right')]
		cdf2we = cwei2[np.searchsorted(data2, data, side='right')]
		d = np.max(np.abs(cdf1we - cdf2we))
		# calculate p-value
		n1 = data1.shape[0]
		n2 = data2.shape[0]
		m, n = sorted([float(n1), float(n2)], reverse=True)
		en = m * n / (m + n)
		if alternative == 'two-sided':
			prob = distributions.kstwo.sf(d, np.round(en))
		else:
			z = np.sqrt(en) * d
			# Use Hodges' suggested approximation Eqn 5.3
			# Requires m to be the larger of (n1, n2)
			expt = -2 * z**2 - 2 * z * (m + 2*n)/np.sqrt(m*n*(m+n))/3.0
			prob = np.exp(expt)
		return d, prob
	
	def calc_pairwise_matrix(self, par_index):
		n_runs = len(self.all_runs)
		combos = itertools.combinations(range(n_runs), 2)
		ks_matrix = np.zeros(shape=(n_runs, n_runs))
		pval_matrix = np.zeros(shape=(n_runs, n_runs))
		for i, j in combos:
			runA = self.all_runs[i]
			runB = self.all_runs[j]
			
			param_samplesA = runA.posterior_samples[:, par_index]
			param_samplesB = runB.posterior_samples[:, par_index]
			ks_stat, pval = self.ks_weighted(param_samplesA, param_samplesB,
											runA.posterior_weights, runB.posterior_weights)
			ks_matrix[j, i] = ks_stat
			ks_matrix[i, j] = ks_stat
			pval_matrix[j, i] = pval
			pval_matrix[i, j] = pval
		return ks_matrix, pval_matrix