#!/bin/zsh

if [ -z "$1" ]
then
    echo "require target model name"
    exit 1
fi
scp "guest_0702@147.46.78.51:NDeepPM/DeepPM/results/$1.pkl" results

