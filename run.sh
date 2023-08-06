# python run_DeepPM.py --data training_data/intel_core.data \
#     --train_cfg config/DeepPM_train.json \
#     --model_cfg config/DeepPM_model.json \
#     --experiment-name pad_zero_baseline \
#     --experiment-time 0702 \
#     --model_class StackedDeepPMPadZero

# python run_DeepPM.py --data training_data/intel_core.data \
#     --train_cfg config/DeepPM_train.json \
#     --model_cfg config/DeepPM_model.json \
#     --experiment-name bi2 \
#     --experiment-time 0702 \
#     --model_class BatchRNN2 \
#     --batch_size 4 \
#     --val_batch_size 4 \
#     --lr 0.5 \
#     --optimizer SGD \
#     --clip_grad_norm 0.1 \
#     --lr_scheduler DecayAfterDelay \
#     # --hyperparameter_test



# TESTING
# 
# 

#  <- NEXT
# python run_DeepPM.py --data training_data/intel_core.data \
#     --train_cfg config/DeepPM_train.json \
#     --model_cfg config/DeepPM_model.json \
#     --experiment-name OSD \
#     --experiment-time 0702 \
#     --model_class OpSrcDest

# python run_DeepPM.py --data training_data/intel_core.data \
#     --train_cfg config/DeepPM_train.json \
#     --model_cfg config/DeepPM_model.json \
#     --experiment-name ithemal_short_only \
#     --experiment-time 0702 \
#     --model_class BatchRNN2 \
#     --batch_size 16 \
#     --val_batch_size 16 \
#     --lr 0.5 \
#     --optimizer SGD \
#     --clip_grad_norm 0.1 \
#     --lr_scheduler DecayAfterDelay \
#     --short_only \
#     --exp_override

python run_DeepPM.py --data training_data/intel_core.data \
    --train_cfg config/DeepPM_train.json \
    --model_cfg config/DeepPM_model.json \
    --experiment-name 044 \
    --experiment-time 0702 \
    --model_class StackedDeepPM044 \
    --short_only

# python run_DeepPM.py --data training_data/intel_core.data \
#     --train_cfg config/DeepPM_train.json \
#     --model_cfg config/DeepPM_model.json \
#     --experiment-name batch_deeppm \
#     --experiment-time 0702 \
#     --model_class DeepPMOriginal \
#     --use_batch_step_lr \
#     --lr 0.0001 \
#     --batch_size 16 \
#     --lr_scheduler OneThirdLR \
#     --exp_override

# python run_DeepPM.py --data training_data/intel_core.data \
#     --train_cfg config/DeepPM_train.json \
#     --model_cfg config/DeepPM_model.json \
#     --experiment-name 080 \
#     --experiment-time 0702 \
#     --model_class StackedDeepPM080

# python run_DeepPM.py --data training_data/intel_core.data \
#     --train_cfg config/DeepPM_train.json \
#     --model_cfg config/DeepPM_model.json \
#     --experiment-name 404 \
#     --experiment-time 0702 \
#     --model_class StackedDeepPM404

# python run_DeepPM.py --data training_data/intel_core.data \
#     --train_cfg config/DeepPM_train.json \
#     --model_cfg config/DeepPM_model.json \
#     --experiment-name 044 \
#     --experiment-time 0702 \
#     --model_class StackedDeepPM044




# python run_DeepPM.py --data training_data/intel_core.data \
#     --train_cfg config/DeepPM_train.json \
#     --model_cfg config/DeepPM_model.json \
#     --experiment-name 440 \
#     --experiment-time 0702 \
#     --model_class StackedDeepPM440









# python run_DeepPM.py --data training_data/intel_core.data \
#     --train_cfg config/DeepPM_train.json \
#     --model_cfg config/DeepPM_model.json \
#     --experiment-name stacked_10ep_000001 \
#     --experiment-time 0702 \
#     --model_class StackedDeepPM \
#     --n_epochs 10 \
#     --lr 0.000001

# python run_DeepPM.py --data training_data/intel_core.data \
#     --train_cfg config/DeepPM_train.json \
#     --model_cfg config/DeepPM_model.json \
#     --experiment-name stacked_10ep_0001 \
#     --experiment-time 0702 \
#     --model_class StackedDeepPM \
#     --n_epochs 10 \
#     --lr 0.0001



