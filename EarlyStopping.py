import numpy as np 
import torch
from pathlib import Path
import pickle 

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, output_dir=None ,patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, optuna_flag=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.optuna_flag = optuna_flag
        self.output_dir = output_dir

    def __call__(self, val_loss, model, plot_val_npmi=None, plot_repr_loss=None, plot_std_loss=None, plot_cov_loss=None, plot_cl_loss=None):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            if plot_val_npmi is not None:
                with open(Path.cwd().joinpath(self.output_dir, 'plot_val_npmi.pkl'),'wb') as f:
                    pickle.dump(plot_val_npmi,f)
            if plot_repr_loss is not None:
                with open(Path.cwd().joinpath(self.output_dir, 'plot_repr_loss.pkl'),'wb') as f:
                    pickle.dump(plot_repr_loss,f)
            if plot_std_loss is not None:
                with open(Path.cwd().joinpath(self.output_dir, 'plot_std_loss.pkl'),'wb') as f:
                    pickle.dump(plot_std_loss,f)
            if plot_cov_loss is not None:
                with open(Path.cwd().joinpath(self.output_dir, 'plot_cov_loss.pkl'),'wb') as f:
                    pickle.dump(plot_cov_loss,f)  
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            if plot_val_npmi is not None:
                with open(Path.cwd().joinpath(self.output_dir, 'plot_val_npmi.pkl'),'wb') as f:
                    pickle.dump(plot_val_npmi,f)
            if plot_repr_loss is not None:
                with open(Path.cwd().joinpath(self.output_dir, 'plot_repr_loss.pkl'),'wb') as f:
                    pickle.dump(plot_repr_loss,f)
            if plot_std_loss is not None:
                with open(Path.cwd().joinpath(self.output_dir, 'plot_std_loss.pkl'),'wb') as f:
                    pickle.dump(plot_std_loss,f)
            if plot_cov_loss is not None:
                with open(Path.cwd().joinpath(self.output_dir, 'plot_cov_loss.pkl'),'wb') as f:
                    pickle.dump(plot_cov_loss,f) 
            if plot_cl_loss is not None:
                with open(Path.cwd().joinpath(self.output_dir, 'plot_cl_loss.pkl'),'wb') as f:
                    pickle.dump(plot_cl_loss,f) 
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss (NPMI) decrease.'''
        if self.verbose:
            # self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.trace_func(f'NPMI increased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        # print(self.optuna_flag)
        if self.optuna_flag == 0:
            with open(self.path, 'wb') as f:
                torch.save(model, f)