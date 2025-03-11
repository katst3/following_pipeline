# handles metrics logging and model artifact uploads on Neptune 

import neptune

class ExperimentTracker:
    """Unified experiment tracking for both model types."""
    
    def __init__(self, project_name, api_token):
        self.project_name = project_name
        self.api_token = api_token
        self.run = None
    
    def start_run(self, run_name, parameters, tags=None):
        """Start a new experiment run."""
        self.run = neptune.init_run(
            project=self.project_name,
            api_token=self.api_token,
            name=run_name,
            tags=tags or []
        )
        
        # log parameters
        self.run["parameters"] = parameters
        
        return self.run
    
    def log_metric(self, metric_name, value, step=None):
        """Log a single metric value."""
        if step is not None:
            self.run[metric_name].append(value, step=step)
        else:
            self.run[metric_name].append(value)
    
    def log_metrics(self, metrics_dict, prefix="", step=None):
        """Log multiple metrics from a dictionary."""
        for metric_name, value in metrics_dict.items():
            full_name = f"{prefix}/{metric_name}" if prefix else metric_name
            self.log_metric(full_name, value, step)
    
    def log_model(self, model_path, model_name):
        """Log a model artifact."""
        self.run[f"models/{model_name}"].upload(model_path)
    
    def end_run(self):
        """End the current experiment run."""
        if self.run is not None:
            self.run.stop()
            self.run = None