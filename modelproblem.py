import os
from pprint import pprint
import petab
import pypesto.petab
from petab.v1.parameters import get_priors_from_df
import benchmark_models_petab as models
from scipy.stats import uniform, norm


class ModelProblem():
	def __init__(self, model_string):
		self.model_name = model_string

	def initialize(self):
		model_name =  self.model_name
		print(f"The location of the model directory is: {models.MODELS_DIR}")
		# the yaml configuration file links to all needed files
		petab_yaml = os.path.join(models.MODELS_DIR, model_name, model_name + ".yaml")

		petab_problem = petab.v1.Problem.from_yaml(petab_yaml)
		importer = pypesto.petab.PetabImporter(petab_problem)

		model = importer.create_model(verbose=False)
		obj = importer.create_objective()
		obj.amici_solver.setRelativeTolerance(1e-8)
		obj.amici_solver.setAbsoluteTolerance(1e-12)
		problem = importer.create_problem(obj, startpoint_kwargs={"check_fval": True})
	
		self.model = model
		self.problem = problem
		self.petab_problem = petab_problem
		"""self.obj = obj

		ret = obj(
			petab_problem.x_nominal_scaled,
			mode="mode_fun",
			return_dict=True,
		)
		pprint(ret); 
		"""
		prior_info = get_priors_from_df(petab_problem.parameter_df,
								  mode="objective")
		self.prior_info = prior_info
		self.n_dim = len(prior_info)
		self.bounds = [x[3] for x in prior_info]
		self.n_fun_calls = 0

	def create_poco_priors(self):
		prior_list = []

		# the list returned from get_priors_from_df is always
		# in the following order:
		# 1) prior type (string)
		# 2) prior parameters (tuple)
		# 3) parameter scale (string - log or linear)
		# 4) parameter bounds (tuple)
		# ignore #3 and #4 
		for info in self.prior_info:
			type, prior_pars, _, _ = info
			if "uniform" in type.lower():
				lb, ub = prior_pars
				prior = uniform(loc=lb, scale=ub-lb)
			elif "normal" in type.lower():
				mean, std = prior_pars
				prior = norm(loc=mean, scale=std)
			prior_list.append(prior)
		return prior_list


	def log_likelihood_wrapper(self, x, mode="pos"):
		try:
			result = self.problem.objective(x, mode="mode_fun", return_dict=True)
			fval = result["fval"]
		except:
			fval = 1e10

		if mode == "neg":
			fval = -1*fval
		# ! IMPORTANT: self.n_fun_calls only tracks total number of function calls
		# ! when using PT-MCMC since it does NOT run in parallel
		# ! You can only use this with pocoMC when n_cpus = 1
		self.n_fun_calls += 1
		return fval