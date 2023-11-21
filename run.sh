# python run_DeepPM.py --cfg config/lstm.yaml --exp_name 'LSTM(256 dim + 1 Layers)' --exp_override 
# python run_DeepPM.py --cfg config/lstm-nt.yaml --exp_name 'LSTM+NT(256 dim + 1 Layer)' --exp_override

# python run_DeepPM.py --cfg config/transformer.yaml --exp_name 'Transformer' --exp_override 
# python run_DeepPM.py --cfg config/transformer-nt.yaml --exp_name 'Transformer+NT' --exp_override

# python run_DeepPM.py --cfg config/transformer-3e.yaml --exp_name 'Transformer+3E' --exp_override
# python run_DeepPM.py --cfg config/transformer-3e-nt.yaml --exp_name 'Transformer+3E+NT' --exp_override 

# python run_DeepPM.py --cfg config/transformer-nn.yaml --exp_name 'Transformer+NN' --exp_override 
# python run_DeepPM.py --cfg config/transformer-nn-nt.yaml --exp_name 'Transformer+NN+NT' --exp_override 

# python run_DeepPM.py --cfg config/transformer-wa.yaml --exp_name 'Transformer+WA' --exp_override 
# python run_DeepPM.py --cfg config/transformer-wa-nt.yaml --exp_name 'Transformer+WA+NT' --exp_override

# python run_DeepPM.py --cfg config/deeppm.yaml --exp_name 'DeepPM+dropout-0.1' --exp_override 
# python run_DeepPM.py --cfg config/deeppm-nt.yaml --exp_name 'DeepPM+NT' --exp_override 

# python run_DeepPM.py --cfg config/deeppm-mod.yaml --exp_name 'DeepPM+Mod' --exp_override 


# python run_DeepPM.py --cfg config/deeppm.yaml --exp_name 'DeepPM+Batch 4' --exp_override  
# python run_DeepPM.py --cfg config/deeppm-sgd.yaml --exp_name 'DeepPM + SGD' --exp_override 
# python run_DeepPM.py --cfg config/deeppm-betas.yaml --exp_name 'DeepPM + Adam betas' --exp_override 
# python run_DeepPM.py --cfg config/deeppm-n_heads-16.yaml --exp_name 'DeepPM + n heads 16' --exp_override 
# python run_DeepPM.py --cfg config/deeppm-relu.yaml --exp_name 'DeepPM + relu' --exp_override 
# python run_DeepPM.py --cfg config/deeppm-dropout.yaml --exp_name 'DeepPM + dropout' --exp_override 

# python run_DeepPM.py --cfg config/deeppm-nt-adamw.yaml --exp_name 'DeepPM+NT+AdamW' --exp_override  
# python run_DeepPM.py --cfg config/deeppm-nt.yaml --exp_name 'DeepPM+NT+fixed lr scheduler' --exp_override  
# python run_DeepPM.py --cfg config/deeppm-nt.yaml --exp_name 'DeepPM+NT+Batch 4' --exp_override  
# python run_DeepPM.py --cfg config/deeppm-nt-sgd.yaml --exp_name 'DeepPM+NT+SGD' --exp_override 
# python run_DeepPM.py --cfg config/deeppm-nt-betas.yaml --exp_name 'DeepPM+NT+Adam betas' --exp_override 
# python run_DeepPM.py --cfg config/deeppm-nt-n_heads-16.yaml --exp_name 'DeepPM+NT+N heads 16' --exp_override 
# python run_DeepPM.py --cfg config/deeppm-nt-relu.yaml --exp_name 'DeepPM+NT+relu' --exp_override 
# python run_DeepPM.py --cfg config/deeppm-nt-dropout.yaml --exp_name 'DeepPM+NT+dropout' --exp_override 

python run_DeepPM.py --cfg config/power-deeppm-nt.yaml --exp_name 'Power+DeepPM+NT+Mapex100' --exp_override 
# python run_DeepPM.py --cfg config/power-lstm-nt.yaml --exp_name 'Power+LSTM+NT+Mapex100' --exp_override 

python run_DeepPM.py --cfg config/power-deeppm.yaml --exp_name 'Power+DeepPM+Mapex100' --exp_override 
# python run_DeepPM.py --cfg config/power-lstm.yaml --exp_name 'Power+LSTM+Mapex100' --exp_override 
