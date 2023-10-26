#python run_DeepPM.py --cfg config/lstm.yaml --exp_name 'LSTM' --exp_override 
# python run_DeepPM.py --cfg config/lstm-nt.yaml --exp_name 'LSTM+NT' --exp_override

# python run_DeepPM.py --cfg config/transformer.yaml --exp_name 'Transformer' --exp_override 
# python run_DeepPM.py --cfg config/transformer-nt.yaml --exp_name 'Transformer+NT' --exp_override

# python run_DeepPM.py --cfg config/transformer-3e.yaml --exp_name 'Transformer+3E' --exp_override
# python run_DeepPM.py --cfg config/transformer-3e-nt.yaml --exp_name 'Transformer+3E+NT' --exp_override 

# python run_DeepPM.py --cfg config/transformer-nn.yaml --exp_name 'Transformer+NN' --exp_override 
# python run_DeepPM.py --cfg config/transformer-nn-nt.yaml --exp_name 'Transformer+NN+NT' --exp_override 

# python run_DeepPM.py --cfg config/transformer-wa.yaml --exp_name 'Transformer+WA' --exp_override 
# python run_DeepPM.py --cfg config/transformer-wa-nt.yaml --exp_name 'Transformer+WA+NT' --exp_override

# python run_DeepPM.py --cfg config/deeppm.yaml --exp_name 'DeepPM' --exp_override 
# python run_DeepPM.py --cfg config/deeppm-nt.yaml --exp_name 'DeepPM+NT' --exp_override 

#python run_DeepPM.py --cfg config/js_lstm.yaml --exp_name 'LSTM' --exp_override
#python run_DeepPM.py --cfg config/js_deeppm.yaml --exp_name 'DeepPM' --exp_override 
python run_DeepPM.py --cfg  js_deeppm_batch32 --exp_name 'DeepPM_batch32_vessl' --exp_override

