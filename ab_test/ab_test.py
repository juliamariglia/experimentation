# libraries
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import beta
from scipy import stats

class ab_aggregate:
  """
  Parameters:

  control_conversions: int
  variant_conversions: int
  control_n: int
  control_n: int
  """
  def __init__(self, control_conversions, variant_conversions, control_n, variant_n):
    self.control_conversions = control_conversions
    self.variant_conversions = variant_conversions
    self.control_n = control_n
    self.variant_n = variant_n
    self.control_ctr = self.control_conversions/self.control_n
    self.variant_ctr = self.variant_conversions/self.variant_n
    self.relative_uplift = (self.variant_ctr - self.control_ctr)/self.control_ctr

  def frequentist(self, one_tailed=True):
    pooled_conversions = ( (self.control_conversions) + (self.variant_conversions) ) \
      / ( (self.control_n) + (self.variant_n) )
      
    # Z represents the number of standard deviations the observed difference of means is away from 0.
    # The higher this number, the lesser the likelihood of H₀
    z = (( self.variant_conversions / self.variant_n ) - ( self.control_conversions / self.control_n )) \
      / np.sqrt( pooled_conversions * (1 - pooled_conversions) * ((1/self.control_n) + (1/self.variant_n)))
    
    se_control = (self.control_ctr*(1-self.control_ctr)/self.control_n)
    se_variant = (self.variant_ctr*(1-self.variant_ctr)/self.control_n)
    ci_95_lower = (self.variant_ctr - self.control_ctr) - 1.96*(np.sqrt(se_control + se_variant))
    ci_95_upper = (self.variant_ctr - self.control_ctr) + 1.96*(np.sqrt(se_control + se_variant))

    if one_tailed == True:
      p = (1 - stats.norm.cdf(z))
    else:
      p = 2 * (1 - stats.norm.cdf(z))
    
    return print('Control CTR:', 
                    '{0:.4f}'.format(self.control_ctr*100), '%',
                '\nVariant CTR:',
                    '{0:.4f}'.format(self.variant_ctr*100), '%',
                '\nRelative Uplift:',
                    '{0:.2f}'.format(self.relative_uplift*100), '%',
                '\nz-statistic:',
                    '{0:.4f}'.format(z), 
                '\np-value:',
                    '{0:.4f}'.format(p), 
                '\nConversion Rate Difference (variant-to-control):',
                    '{0:.2f}'.format((self.variant_ctr - self.control_ctr)*100), '%',
                '\nConfidence Interval for Conversion Rate Difference (95%): ', 
                    '[ {0:.4f}'.format(ci_95_lower*100), '% - ', '{0:.4f}'.format((ci_95_upper*100)), '% ]')

  def bayesian(self, prior_success = 0, prior_failure = 0, n_simulation = 1000):
    # setting prior parameters
    prior_alpha = prior_success + 1
    prior_beta = prior_failure + 1

    if prior_success==0 | prior_failure==0:
        prior_ctr=None
    else:
        prior_ctr = '{0:.4f}'.format((prior_success/(prior_success + prior_failure))*100)

    control_failure = self.control_n - self.control_conversions
    variant_failure = self.variant_n - self.variant_conversions

    posterior_control = beta(prior_alpha+self.control_conversions, prior_beta+control_failure)
    posterior_variant = beta(prior_alpha+self.variant_conversions, prior_beta+variant_failure)

    beta_prior_distribution = beta(prior_alpha, prior_beta)
    x = np.linspace(self.control_ctr - self.control_ctr*0.1, self.control_ctr*1.1, 1000)
    plt.plot(x, beta_prior_distribution.pdf(x), label=f'prior({prior_alpha}, {prior_beta})', c = 'blue') # plot priors
    plt.plot(x, posterior_control.pdf(x), label='control', c = 'green')
    plt.plot(x, posterior_variant.pdf(x), label='treatment', c='orange')
    plt.xlabel('Conversion Rate (fraction)')
    plt.ylabel('Density')
    plt.title('Prior (blue) & Experiment Posteriors')
    plt.legend()
    plt.show()

    # simulation to get probability of variant to be better than control
    sim_posterior_control = np.random.beta(prior_alpha+self.control_conversions, prior_beta+control_failure, size=n_simulation)
    sim_posterior_variant = np.random.beta(prior_alpha+self.variant_conversions, prior_beta+variant_failure, size=n_simulation)
    probability_variant_is_better = np.mean(sim_posterior_control < sim_posterior_variant)*100

    return print('Control CTR (%):', '{0:.4f}'.format(self.control_ctr*100),
                '\nVariant CTR (%):', '{0:.4f}'.format(self.variant_ctr*100),
                '\nRelative Uplift (%):', '{0:.2f}'.format(self.relative_uplift*100),
                '\n'
                '\nModeled Posterior Control CTR (%):', '{0:.4f}'.format(sim_posterior_control.mean()*100),
                '\nModeled Posterior Variant CTR (%):', '{0:.4f}'.format(sim_posterior_variant.mean()*100),
                '\nPrior CTR (%):', prior_ctr,
                '\nProbability variant is better than control (%):', probability_variant_is_better)
                
class ab_accumulating:
    """
    Parameters:

    control_conversions: (n,) array of daily control conversions
    variant_conversions: (n,) array of daily variant conversions
    control_n: (n,) array of daily control counts
    control_n: (n,) array of daily variant counts
    """

    def __init__(self, control_conversions, variant_conversions, control_n, variant_n):
        self.control_conversions = control_conversions
        self.variant_conversions = variant_conversions
        self.control_n = control_n
        self.variant_n = variant_n
        self.control_ctr = self.control_conversions/self.control_n
        self.variant_ctr = self.variant_conversions/self.variant_n
        self.relative_uplift = (self.variant_ctr - self.control_ctr)/self.control_ctr

    def bayesian(self, prior_success = 0, prior_failure = 0, n_simulation = 1000, n_days = None):
        # setting priors
        prior_alpha = prior_success + 1
        prior_beta = prior_failure + 1

        control_failure = self.control_n - self.control_conversions
        variant_failure = self.variant_n - self.variant_conversions

        if n_days == None:
            days = len(self.control_n)
        else:
            days = n_days
            
        probability_variant_is_better_append = []
        days_append = []

        for day in range(1, days+1):
            days_append.append(day)
            sim_posterior_control = np.random.beta(
              prior_alpha+self.control_conversions[:day].sum(),
              prior_beta+control_failure[:day].sum(), size=n_simulation)
            
            sim_posterior_variant = np.random.beta(
              prior_alpha+self.variant_conversions[:day].sum(),
              prior_beta+variant_failure[:day].sum(), size=n_simulation)
            
            probability_variant_is_better = np.mean(sim_posterior_control < sim_posterior_variant)*100
            probability_variant_is_better_append.append(probability_variant_is_better)
        
        plt.plot(days_append, probability_variant_is_better_append)
        plt.title('Probability of variant to be better than control')
        plt.xlabel('Experiment Duration (days)')
        plt.ylabel('Probability (%)')
        plt.hlines(95, 0, days, color='red', label='Threshold (95 %)')
        plt.legend()
        plt.show()
        
        return days_append, probability_variant_is_better_append