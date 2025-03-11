# wrapper for Ax hyperparameter optimization 

from ax.service.ax_client import AxClient

class HyperparameterOptimizer:
    
    def __init__(self, param_space, objective_name="val_accuracy", minimize=False):
        self.param_space = param_space
        self.objective_name = objective_name
        self.minimize = minimize
        self.client = AxClient()
        
        # set up parameter space
        parameters = []
        for param in param_space:
            # handle different parameter types
            if param['type'] == 'range':
                bounds = param['bounds']
                parameters.append({
                    'name': param['name'],
                    'type': 'range',
                    'bounds': bounds,
                    'log_scale': param.get('log_scale', False)
                })
            elif param['type'] == 'choice':
                parameters.append({
                    'name': param['name'],
                    'type': 'choice',
                    'values': param['values']
                })
            elif param['type'] == 'int':
                parameters.append({
                    'name': param['name'],
                    'type': 'range',
                    'bounds': param['bounds'],
                    'value_type': 'int'
                })
        
        # create experiment
        self.client.create_experiment(
            name="model_hyperparameter_optimization",
            parameters=parameters,
            objective_name=objective_name,
            minimize=minimize
        )
    
    def get_next_parameters(self):
        """Get next set of parameters to try."""
        parameters, trial_index = self.client.get_next_trial()
        return parameters, trial_index
    
    def complete_trial(self, trial_index, result):
        """Complete a trial with the result."""
        self.client.complete_trial(trial_index=trial_index, raw_data=result)
    
    def get_best_parameters(self):
        """Get best parameters found so far."""
        return self.client.get_best_parameters()