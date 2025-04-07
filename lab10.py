import numpy as np
import scipy.stats as stat
import pandas as pd
from scipy.optimize import minimize

class RegressionModel(object):
    def __init__(self, x, y, create_intercept=True, regression_type='ols'):
        if isinstance(x, pd.DataFrame):
            self.x = x
        else:
            raise RuntimeError("Matrix 'x' is not a DataFrame.")
        if isinstance(y, pd.DataFrame) | isinstance(y, pd.Series):
            self.y = y
        else:
            raise RuntimeError("Matrix 'y' is not a DataFrame.")
        if isinstance(create_intercept, bool):
            self.create_intercept = create_intercept
            if self.create_intercept:
                self.add_intercept()
        else:
            raise RuntimeError("Parameter 'create_intercept' must be a boolean value.")
        if isinstance(regression_type, str):
            if regression_type=="ols":
                self.regression_type=regression_type
            elif regression_type=='logit':
                self.regression_type=regression_type
            else:
                raise RuntimeError("Only OLS and Logistic regressions ('ols' or 'logit', respectively) are supported")
        else:
            raise RuntimeError("Parameter 'regression_type' must be a string with value 'ols' or 'logit'.")
            
    def add_intercept(self):
        self.x = self.x.assign(intercept=pd.Series([1]*np.shape(self.x)[0]))



    def sigmoid(self,z):
      # the z represents the linear combination of the input features and their coefficients (betas)
      # z=X @ β
      return 1 / (1 + np.exp(-z))   

    def likelihood(self, beta, x, y):
      beta = beta.flatten()
      z = x @ beta
      p = self.sigmoid(z)

      # add to avoid log(0)
      epsilon = 1e-9
      log_likelihood =  -1 * np.sum(y * np.log(p + epsilon) + (1 - y) * np.log(1 - p + epsilon))
      return log_likelihood

    def gradient(self, beta,x,y):
      beta = beta.flatten()
      z = x @ beta
      p = self.sigmoid(z)
      return -1 * x.T @ (y - p)
    
    # def true_beta(self, learning_rate=0.01, max_iter=100, tol=1e-6):
    #   n, k = np.shape(self.x)
    #   beta = np.zeros((k, 1))
    #   for i in range(max_iter):
    #     grad = self.gradient(beta)
    #     beta_new = beta + learning_rate * grad

    #     if np.linalg.norm(beta_new - beta, ord=1) < tol:
    #         break
    #     beta = beta_new
    #   return beta

    def compute_se_logit(self, beta):
      z = self.x @ beta
      p = self.sigmoid(z).to_numpy()
      w = np.diag((p * (1 - p)).flatten())
      Fisher_info = self.x.T @ w @ self.x
      cov_matrix = np.linalg.inv(Fisher_info)
      se = np.sqrt(np.diag(cov_matrix)).reshape(-1, 1)
      return se.flatten()


    def compute_se_logit2(self, beta):
        beta = beta.flatten()
        x = np.asarray(self.x)
        z = x @ beta
        p = self.sigmoid(z)

        # Numerical stability
        eps = 1e-9
        p = np.clip(p, eps, 1 - eps)

        W = np.diag(p * (1 - p))
        Fisher_info = x.T @ W @ x
        cov_matrix = np.linalg.inv(Fisher_info)

        se = np.sqrt(np.diag(cov_matrix)).reshape(-1, 1)
        return se.flatten()



    def compute_se_ols(self,beta):
      x = self.x
      y = self.y
      n, k = np.shape(x)
      eps = y - x.dot(beta)
      shat = eps.T.dot(eps)/(n-k)
      covar = shat * np.linalg.solve(x.T.dot(x), np.eye(k))
      var = np.diag(covar)
      se = np.asarray([np.sqrt(i) for i in var])
      return se

    def ols_regression(self):
      x = self.x
      y = self.y
      n, k = np.shape(x)
      beta = np.dot(np.linalg.solve(x.T.dot(x), np.eye(k)), x.T.dot(y))
      serror = self.compute_se_ols(beta)
      tstat = np.asarray([i[1]/serror[i[0]] for i in enumerate(beta)])
      pval = stat.t.sf(tstat, n-k)

      self.results = dict()
      for i in enumerate(self.x.columns):
        self.results[i[1]] = {
                'coefficient' : beta[i[0]],
                'standard_error' : serror[i[0]],
                't_stat' : tstat[i[0]],
                'p_value' : pval[i[0]]
                }

    def logit_regression(self):
      x = self.x
      y = self.y
      n, k = np.shape(x)
      init_beta = np.zeros(k)
      result = minimize(self.likelihood, init_beta, args=(x.values, y.values), method='BFGS', jac=self.gradient, options={'disp': True, 'maxiter': 10000})
      beta = result.x
      serror = self.compute_se_logit(beta)
      z_stat = beta / serror.flatten()
      pval = 2 * (1 - stat.norm.cdf(np.abs(z_stat)))

      self.results = dict()
      for i in enumerate(self.x.columns):
          self.results[i[1]] = {
                  'coefficient' : beta[i[0]],
                  'standard_error' : serror[i[0]],
                  'z_stat' : z_stat[i[0]],
                  'p_value' : pval[i[0]]
                  }
      
    def fit_model(self):
      if self.regression_type=="ols":
        self.ols_regression()
      elif self.regression_type=='logit':
        self.logit_regression()
      else:
        raise RuntimeError("Only OLS and Logistic regressions ('ols' or 'logit', respectively) are supported")


    def summary(self):
        print("Variable Name".ljust(25) + "| Coefficient".ljust(15) + "| Standard Error".ljust(17) + "| Z-Statistic".ljust(15) + "| P-Value".ljust(15) + "\n" + "-"*85)
        for i in self.results:
            print("{}".format(i).ljust(25) + "| {}".format(round(self.results[i]['coefficient'], 3)).ljust(15) + "| {}".format(round(self.results[i]['standard_error'], 3)).ljust(17) + "| {}".format(round(self.results[i]['z_stat'], 3)).ljust(15) + "| {}".format(round(self.results[i]['p_value'], 3)).ljust(15))
