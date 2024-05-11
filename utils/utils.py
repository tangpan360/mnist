import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


class EarlyStopping(object):

    def __init__(self, patience=500, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, filepath):

        val_loss = val_loss

        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path, filepath)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path, filepath)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, filepath):

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        for name in os.listdir(filepath):
            file = './' + filepath + '/' + name
            os.remove(file)

        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
