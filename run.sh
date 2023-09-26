# python run_DeepPM.py --cfg config/transformer_224.yaml --exp_name 'Tokenizer + Transformer_224' --exp_override

# python run_DeepPM.py --cfg config/batch_rnn.yaml --exp_name 'Tokenizer + Ithemal' --exp_override

# python run_DeepPM.py --cfg config/bert_baseline.yaml --exp_name 'Bert_baseline' --exp_override

# python run_DeepPM.py --cfg config/transformer_only.yaml --exp_name 'Tokenizer + Transformer_only' --exp_override

# python run_DeepPM.py --cfg config/deeppm.yaml --exp_name 'Tokenizer + DeepPM' --exp_override


python run_DeepPM.py --cfg config/deeppm_param_searching.yaml --exp_name 'Tokenizer + DeepPM Fixed(SGD)' --exp_override
