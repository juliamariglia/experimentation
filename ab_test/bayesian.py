import numpy as np
from typing import final
from scipy.stats import beta, gamma

class bayesian_conversion_test:
    """
    AB conversion and numeric test for two samples.
    Parameters:
    simulation_size: int
    """
    def __init__(self, simulation_size=10000, set_seed=9):
        
        self.simulation_size = simulation_size
        self.set_seed=set_seed
        
    def evaluate(self, A_conversions=None, B_conversions=None, A_prior_beta=None, B_prior_beta=None,
                       prior_success=0, prior_failure=0):
        """
        Binomial likelihood with weak Beta priors. A is control, B is treatment.
        Parameters:
        A_conversions, B_conversions: user conversions, array
        """
        
        # derived from input
        A_sum_conversions = A_conversions.sum()
        B_sum_conversions = B_conversions.sum()
        A_n = len(A_conversions)
        B_n = len(B_conversions)
        
        self.A_conversion_rate = A_sum_conversions/A_n
        self.B_conversion_rate = B_sum_conversions/B_n
        self.relative_uplift = (self.B_conversion_rate - self.A_conversion_rate)/self.A_conversion_rate
        
        # setting prior parameters
        prior_alpha = prior_success + 1
        prior_beta = prior_failure + 1

        if prior_success==0 | prior_failure==0:
            self.prior_conversion_rate=None
        else:
            self.prior_conversion_rate = '{0:.4f}'.format((prior_success/(prior_success + prior_failure))*100)
            
        A_failure = A_n - A_sum_conversions
        B_failure = B_n - B_sum_conversions

        posterior_A = beta(prior_alpha+A_sum_conversions, prior_beta+A_failure)
        posterior_B = beta(prior_alpha+B_sum_conversions, prior_beta+B_failure)
        
        np.random.seed(self.set_seed)
        
        # simulation of beta distribution for posterior
        self.sim_posterior_A = np.random.beta(prior_alpha+A_sum_conversions, prior_beta+A_failure, size=self.simulation_size)
        self.sim_posterior_B = np.random.beta(prior_alpha+B_sum_conversions, prior_beta+B_failure, size=self.simulation_size)
        self.probability_B_is_better_than_A = np.mean(self.sim_posterior_A < self.sim_posterior_B)*100
        self.probability_A_is_better_than_B = np.mean(self.sim_posterior_A > self.sim_posterior_B)*100
        
        return posterior_A, posterior_B, self.sim_posterior_A, self.sim_posterior_B
    
    @final
    def describe(self):
        return print('A Conversion Rate (%):', '{0:.4f}'.format(self.A_conversion_rate*100),
                '\nB Conversion Rate (%):', '{0:.4f}'.format(self.B_conversion_rate*100),
                '\nRelative Uplift (%):', '{0:.2f}'.format(self.relative_uplift*100),
                '\n'
                '\nModeled Posterior A Conversion Rate (%):', '{0:.4f}'.format(self.sim_posterior_A.mean()*100),
                '\nModeled Posterior B Conversion Rate (%):', '{0:.4f}'.format(self.sim_posterior_B.mean()*100),
                '\nPrior CTR (%):', self.prior_conversion_rate,
                '\nProbability B is better than A (%):', self.probability_B_is_better_than_A,
                '\nProbability A is better than B (%):', self.probability_A_is_better_than_B)


class bayesian_numeric_test:
    """
    AB conversion and numeric test for two samples.
    Parameters:
    simulation_size: int
    """
    def __init__(self, simulation_size=10000, set_seed=9):
        
        self.simulation_size = simulation_size
        self.set_seed=set_seed
        
    def evaluate(self, A_conversions=None, B_conversions=None, A_metric=None, B_metric=None,
                     A_prior_gamma_alpha=None, B_prior_gamma_alpha=None, A_prior_gamma_scale=None, B_prior_gamma_scale=None):
        """
        Exponential likelihood with weak Gamma priors.
        Parameters:
        A_conversions, B_conversions: array of user conversions, binary
        A_metric, B_metric: array of user metrics, numeric
        [A/B]_prior_gamma_alpha: numeric value representing gamma distributions' alpha parameter, float
        [A/B]_prior_gamma_scale: numeric value representing gamma distributions' scale parameter, float
        """
        
        # derived from input
        A_sum_conversions = A_conversions.sum()
        B_sum_conversions = B_conversions.sum()
        A_sum_metric = A_metric.sum()
        B_sum_metric = B_metric.sum()
        self.A_mean_metric = A_metric.mean()
        self.B_mean_metric = B_metric.mean()
        A_n = len(A_conversions)
        B_n = len(B_conversions)
        
        self.relative_uplift = ((B_metric.mean()/A_metric.mean() - 1)*100).round(2)
        self.absolute_difference = (B_metric.mean() - A_metric.mean()).round(2)
        
        A_prior_lambda = gamma(a=(A_prior_gamma_alpha), scale=(A_prior_gamma_scale))
        B_prior_lambda = gamma(a=(B_prior_gamma_alpha), scale=(B_prior_gamma_scale))

        A_post_lambda = gamma(a=(A_prior_gamma_alpha + A_sum_conversions), scale=(A_prior_gamma_scale/(1 + (A_prior_gamma_scale)*A_sum_metric)))
        B_post_lambda = gamma(a=(B_prior_gamma_alpha + B_sum_conversions), scale=(B_prior_gamma_scale/(1 + (B_prior_gamma_scale)*B_sum_metric)))

        np.random.seed(self.set_seed)
        
        # simulations of expected numeric value through simulations of gamma parameter
        self.A_simulated_metric = 1/np.random.gamma(shape=(A_prior_gamma_alpha + A_sum_conversions), scale=(A_prior_gamma_scale/(1 + (A_prior_gamma_scale)*A_sum_metric)), size = self.simulation_size)
        self.B_simulated_metric = 1/np.random.gamma(shape=(B_prior_gamma_alpha + B_sum_conversions), scale=(B_prior_gamma_scale/(1 + (B_prior_gamma_scale)*B_sum_metric)), size = self.simulation_size)
        
        B_won = [i <= j for i,j in zip(self.A_simulated_metric, self.B_simulated_metric)]
        self.probability_B_is_better_than_A = np.mean(B_won)
        A_won = [i >= j for i,j in zip(self.A_simulated_metric, self.B_simulated_metric)]
        self.probability_A_is_better_than_B = np.mean(A_won)

        return A_post_lambda, B_post_lambda, self.A_simulated_metric, self.B_simulated_metric
    
    @final
    def describe(self):

        return print('Actuals'
                '\nA Avg Metric:', '{0:.4f}'.format(self.A_mean_metric),
                '\nB Avg Metric:', '{0:.4f}'.format(self.B_mean_metric),
                '\nRelative Uplift (%):', '{0:.2f}'.format(self.relative_uplift),
                '\nAbsolute Difference (%):', '{0:.2f}'.format(self.absolute_difference),
                '\n\nModeled'
                '\nModeled Exponential A Value (%):', '{0:.4f}'.format(self.A_simulated_metric.mean()),
                '\nModeled Exponential B Value (%):', '{0:.4f}'.format(self.B_simulated_metric.mean()),
                '\nProbability B is better than A (%):', self.probability_B_is_better_than_A,
                '\nProbability A is better than B (%):', self.probability_A_is_better_than_B)