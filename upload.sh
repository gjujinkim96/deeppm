#!/bin/zsh


scp config/* guest_0702@147.46.78.51:NDeepPM/DeepPM/config
scp data/* guest_0702@147.46.78.51:NDeepPM/DeepPM/data
scp datasets/* guest_0702@147.46.78.51:NDeepPM/DeepPM/datasets
scp losses/* guest_0702@147.46.78.51:NDeepPM/DeepPM/losses
scp lr_schedulers/* guest_0702@147.46.78.51:NDeepPM/DeepPM/lr_schedulers
scp models/* guest_0702@147.46.78.51:NDeepPM/DeepPM/models
scp optimizers/* guest_0702@147.46.78.51:NDeepPM/DeepPM/optimizers

scp * guest_0702@147.46.78.51:NDeepPM/DeepPM

