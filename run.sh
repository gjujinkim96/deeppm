

# python run_DeepPM.py --data training_data/intel_core.data \
#     --train_cfg config/DeepPM_train.json \
#     --model_cfg config/DeepPM_model.json \
#     --experiment-name ithemal \
#     --experiment-time 0702 \
#     --model_class Ithemal \
#     --raw_data 

# python run_DeepPM.py --data training_data/intel_core.data \
#     --train_cfg config/DeepPM_train.json \
#     --model_cfg config/DeepPM_model.json \
#     --experiment-name batch_ithemal_lr:0.1 \
#     --experiment-time 0702 \
#     --lr 0.1 \
#     --batch_size 32 \
#     --val_batch_size 32 \
#     --clip_grad_norm 2 \
#     --lr_scheduler StepLR \
#     --model_class BatchIthemal

# python run_DeepPM.py --data training_data/intel_core.data \
#     --train_cfg config/DeepPM_train.json \
#     --model_cfg config/DeepPM_model.json \
#     --experiment-name bi_fix \
#     --experiment-time 0702 \
#     --model_class BatchIthemal

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

python run_DeepPM.py --data training_data/intel_core.data \
    --train_cfg config/DeepPM_train.json \
    --model_cfg config/DeepPM_model.json \
    --experiment-name 044 \
    --experiment-time 0702 \
    --model_class StackedDeepPM044




# python run_DeepPM.py --data training_data/intel_core.data \
#     --train_cfg config/DeepPM_train.json \
#     --model_cfg config/DeepPM_model.json \
#     --experiment-name instblockop \
#     --experiment-time 0702 \
#     --model_class InstBlockOp

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



