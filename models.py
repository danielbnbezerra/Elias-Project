import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import timeseries
from darts.utils.statistics import check_seasonality, plot_acf
from darts.models import NBEATSModel
from darts.models import NHiTSModel

# class Model:
#     def __init__(self, selected_model):
#         self.model = selected_model()
# NBEATS Model
# def model_run():
#     if app.toplevel
# model = NBEATSModel(
#     input_chunk_length=study.best_trial.params['in_len'],
#     output_chunk_length=study.best_trial.params['out_len'],
#     batch_size=2**study.best_trial.params['batch_size'],
#     layer_widths=2**study.best_trial.params['layer_widths'],
#     dropout=study.best_trial.params['dropout'],
#     num_stacks=study.best_trial.params['num_stacks'],
#     num_blocks=study.best_trial.params['num_blocks'],
#     num_layers=study.best_trial.params['num_layers'],
#     n_epochs=n_epochs,
#     nr_epochs_val_period=1,
#     optimizer_kwargs={"lr": study.best_trial.params['lr']},
#     pl_trainer_kwargs=set_pl_trainer_kwargs(),#(callbacks = [EarlyStopping("val_loss", min_delta=min_delta, patience=patience, verbose=True)]),
#     save_checkpoints=True,
#     random_state = 0
# )
# model.fit(train, val_series=val)
# eval_model(model, len(val_scaled) + 48, series_scaled, val_scaled)
# print('best_model')
# model = NBEATSModel.load_from_checkpoint(model.model_name)
# eval_model(model, len(val_scaled) + 48, series_scaled, val_scaled)