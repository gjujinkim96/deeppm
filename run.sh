# python run_DeepPM.py --cfg config/bert_baseline.yaml --exp_name retest 


# python run_DeepPM.py --cfg config/batch_rnn.yaml --exp_name rnn_test

# Try later
# python run_DeepPM.py --cfg config/transformer_only.yaml --exp_name Transformers 

# python run_DeepPM.py --cfg config/baseline.yaml --exp_name 'Tokenizer + 224_Layer_betas_0.99'

# python run_DeepPM.py --cfg config/no_layernorm_224.yaml --exp_name 'Tokenizer + 224_Layer_no_layernorm' --exp_override

# python run_DeepPM.py --cfg config/deeppm_learnable_dist_w.yaml --exp_name 'Tokenizer + DeepPM Learnable DW' --exp_override

python run_DeepPM.py --cfg config/deeppm.yaml --exp_name 'Tokenizer + DeepPM Fixed' --exp_override
