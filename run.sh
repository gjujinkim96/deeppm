# python run_DeepPM.py --cfg config/bert_baseline.yaml --exp_name retest 


# python run_DeepPM.py --cfg config/batch_rnn.yaml --exp_name rnn_test

# Try later
# python run_DeepPM.py --cfg config/transformer_only.yaml --exp_name Transformers 

# python run_DeepPM.py --cfg config/baseline.yaml --exp_name 'Tokenizer + 224_Layer_betas_0.99'

python run_DeepPM.py --cfg config/deeppm.yaml --exp_name 'Tokenizer + DeepPM' --exp_override
