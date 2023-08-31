# python run_DeepPM.py --cfg config/extra_tags.yaml --exp_name best_small_fix --exp_override
# for training later

# python run_DeepPM.py --cfg config/just_linear.yaml --exp_name just_linear --exp_override
# python run_DeepPM.py --cfg config/bert_pretrain.yaml --exp_name bp_12_layer --exp_override
# python run_DeepPM.py --cfg config/bert_mixed_only.yaml --exp_name bertMixedOnly --exp_override

# new raw_data
# python run_DeepPM.py --cfg config/baseline.yaml --exp_name SR:B --exp_override
python run_DeepPM.py --cfg config/baseline_test6.yaml --exp_name I1:SR:BT6 --exp_override
