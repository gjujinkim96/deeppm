# python run_DeepPM.py --cfg config/extra_tags.yaml --exp_name best_small_fix --exp_override
# for training later

# python run_DeepPM.py --cfg config/just_linear.yaml --exp_name just_linear --exp_override
# python run_DeepPM.py --cfg config/bert_pretrain.yaml --exp_name bp_12_layer --exp_override
# python run_DeepPM.py --cfg config/bert_mixed_only.yaml --exp_name bertMixedOnly --exp_override

# new raw_data
# python run_DeepPM.py --cfg config/baseline.yaml --exp_name SR:B_splited_evenly --exp_override
# python run_DeepPM.py --cfg config/baseline_test6.yaml --exp_name I1:SR:BT6 --exp_override
# python run_DeepPM.py --cfg config/non_stacked.yaml --exp_name I1:SR:ns_large_16_layer --exp_override

# python run_DeepPM.py --cfg config/batch_rnn.yaml --exp_name Stacked:bi2_batch_32_norm_1.0 --exp_override
python run_DeepPM.py --cfg config/inst_block_op.yaml --exp_name SR:inst_block_op --exp_override