import wandb
import torch
import pandas as pd

def wandb_init(args, model_cfg, train_cfg, train_data_len):
    mode = 'disabled' if args.wandb_disabled else 'online'
    run = wandb.init(
        project='deeppm',
        config={
            "model_class": model_cfg.model_class,
            "dim": model_cfg.dim,
	        "dim_ff": model_cfg.dim_ff,
	        "n_layers": model_cfg.n_layers,
	        "n_heads": model_cfg.n_heads,
	        "max_len": model_cfg.max_len,
            "stacked_data": model_cfg.stacked,
            "only_unique": model_cfg.only_unique,

            "seed": train_cfg.seed,
            "batch_size": train_cfg.batch_size,
            "val_batch_size": train_cfg.val_batch_size,
            "lr": train_cfg.lr,
            "lr_total_iters": train_cfg.lr_total_iters,
            "n_epochs": train_cfg.n_epochs,
            "lr_scheduler": train_cfg.lr_scheduler,
            "optimizer": train_cfg.optimizer,
            "loss_fn": train_cfg.loss_fn,
            "checkpoint": train_cfg.checkpoint,
            'raw_data': train_cfg.raw_data,

            "small_size": args.small_size,
            "small_training": args.small_training,

            "steps_per_epoch": (train_data_len + train_cfg.batch_size - 1) // train_cfg.batch_size,
        },
        # settings=wandb.Settings(code_dir="., "),
        mode=mode,
        name=args.experiment_name,
        tags=['sum_zero'],
    )

    run.log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".sh") or path.endswith(".json"))


    wandb.define_metric("loss/Validation", summary="min")
    wandb.define_metric("val/correct/Threshold 25", summary="max")



def wandb_finish():
    wandb.finish()

cat_names = ['A:1~22', 'B:23~49', 'C:50~99', 'D:100~149', 'E:150~199', 'F:200~']

def cat_by_inst(value):
    idx = 0
    if value >= 23:
        idx += 1
    if value >= 50:
        idx += 1
    if value >= 100:
        idx += 1
    if value >= 150:
        idx += 1
    if value >= 200:
        idx += 1

    return cat_names[idx]

def wandb_log_val(er, epoch):
    df = pd.DataFrame.from_dict({
        'predicted': er.prediction,
        'measured': er.measured,
        'inst_lens': er.inst_lens,
    })

    df['mape'] = abs(df.predicted - df.measured) * 100 / (df.measured + 1e-5)

    logging_dict = {
        "loss/Validation": er.loss,
        'epoch': epoch,
    }

    # print(len(er.prediction), len(er.measured))
    scatter_data = [[x, y] for (x, y) in zip(df.predicted, df.measured)]
    # print(data)
    scatter_table = wandb.Table(data=scatter_data, columns = ["Predicted", "Measured"])
    logging_dict["val/summary/scatter"] = wandb.plot.scatter(scatter_table, "Predicted", "Measured", title="Measured vs. Predicted")

    threshold_data = [[i, (df.mape < i).mean()] for i in range(1, 100)]
    threshold_table = wandb.Table(data=threshold_data, columns=["Threshold", "Correct"])
    logging_dict["val/summary/threshold"] = wandb.plot.line(threshold_table, "Threshold", "Correct", title="Threshold vs. Correct")

    thresholds = [25, 20, 15, 10, 5, 3]
    for threshold in thresholds:
        logging_dict[f'val/correct/Threshold {threshold}'] = (df.mape < threshold).mean()


    df['cat'] = pd.Categorical(df.inst_lens.apply(cat_by_inst), ordered=True, categories=cat_names)
    cat_mape_mean = df.groupby('cat').mape.mean()
    for cat_name in cat_names:
        logging_dict[f'val/cat/MAPE Error {cat_name}'] = cat_mape_mean[cat_name]

    mape_data = [[label, val] for (label, val) in zip(cat_names, cat_mape_mean)]
    mape_table = wandb.Table(data=mape_data, columns = ["Number of Instructions", "Mean MAPE Error"])
    logging_dict["val/summary/mean error"] = wandb.plot.bar(mape_table, "Number of Instructions", 'Mean MAPE Error', title="Mean Error Over Size")

    # print(logging_dict)
    thresholds = [25, 20, 15]
    for threshold in thresholds:
        df[f't_{threshold}'] = df.mape < threshold
        cat_correct = df.groupby('cat')[f't_{threshold}'].mean()
        for cat_name in cat_names:
            logging_dict[f'val/correct/threshold_{threshold}/{cat_name}'] = cat_correct[cat_name]

    wandb.log(logging_dict)

def wandb_log_train(br, lr, epoch):
    df = pd.DataFrame.from_dict({
        'predicted': br.prediction,
        'measured': br.measured,
        'inst_lens': br.inst_lens,
    })

    df['mape'] = abs(df.predicted - df.measured) * 100 / (df.measured + 1e-5)

    logging_dict = {
        "loss/Train": br.loss,
        "lr": lr,
        'epoch': epoch,
    }

    thresholds = [25, 20, 15, 10, 5, 3]
    for threshold in thresholds:
        logging_dict[f'train/correct/Threshold {threshold}'] = (df.mape < threshold).mean()

    wandb.log(logging_dict)
