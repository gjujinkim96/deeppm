#!/bin/zsh

scp data/* guest_0702@147.46.78.51:NDeepPM/DeepPM/data
scp config/* guest_0702@147.46.78.51:NDeepPM/DeepPM/config
scp * guest_0702@147.46.78.51:NDeepPM/DeepPM
scp models/* guest_0702@147.46.78.51:NDeepPM/DeepPM/models
