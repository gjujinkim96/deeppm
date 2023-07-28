import wandb
import torch
import pandas as pd

def wandb_init(args, model_cfg, train_cfg, train_data_len):
    mode = 'disabled' if args.wandb_disabled else 'online'
    wandb.init(
        project='deeppm',
        config={
            "model_class": model_cfg.model_class,
            "dim": model_cfg.dim,
	        "dim_ff": model_cfg.dim_ff,
	        "n_layers": model_cfg.n_layers,
	        "n_heads": model_cfg.n_heads,
	        "max_len": model_cfg.max_len,

            "seed": train_cfg.seed,
            "batch_size": train_cfg.batch_size,
            "lr": train_cfg.lr,
            "n_epochs": train_cfg.n_epochs,
            "stacked_data": train_cfg.stacked,
            "lr_scheduler": train_cfg.lr_scheduler,
            "optimizer": train_cfg.optimizer,
            "loss_fn": train_cfg.loss_fn,

            "small_size": args.small_size,
            "small_training": args.small_training,

            "steps_per_epoch": (train_data_len + train_cfg.batch_size - 1) // train_cfg.batch_size,
        },
        mode=mode,
    )

def wandb_finish():
    wandb.finish()

def cat_by_inst(value):
    if value < 23:
        return '1~22'
    elif value < 50:
        return '23~49'
    elif value < 100:
        return '50~99'
    elif value < 150:
        return '100~149'
    elif value < 200:
        return '150~199'
    return '200~'

def wandb_log_val(er):
    df = pd.DataFrame.from_dict({
        'predicted': er.prediction,
        'measured': er.measured,
        'inst_lens': er.inst_lens,
    })

    df['mape'] = abs(df.predicted - df.measured) * 100 / (df.measured + 1e-5)

    logging_dict = {
        "loss/Validation": er.loss,
    }

    data = [[x, y] for (x, y) in zip(er.prediction, er.measured)]
    table = wandb.Table(data=data, columns = ["predicted", "measured"])
    logging_dict["val/summary/Measured vs. Predicted"] = wandb.plot.scatter(table, "Predicted", "Measured", title="Measured vs. Predicted")

    thresholds = [25, 20, 15, 10, 5, 3]
    for threshold in thresholds:
        logging_dict[f'val/correct/Threshold {threshold}'] = (df.mape < threshold).mean()


    cat_names = ['1~22', '23~49', '50~99', '100~149', '150~199', '200~']
    df['cat'] = pd.Categorical(df.inst_lens.apply(cat_by_inst), ordered=True, categories=cat_names)
    cat_mape_mean = df.groupby('cat').mape.mean()
    for cat_name in cat_names:
        logging_dict[f'val/cat/MAPE Error {cat_name}'] = cat_mape_mean[cat_name]

    data = [[label, val] for (label, val) in zip(cat_names, cat_mape_mean)]
    table = wandb.Table(data=data, columns = ["Number of Instructions", "Mean MAPE error"])
    logging_dict["val/summary/Mean Error Over Size"] = wandb.plot.bar(table, "Number of Instructions", "Mean MAPE error", title="Mean Error Over Size")

    for threshold in thresholds:
        df[f't_{threshold}'] = df.mape < threshold
        cat_correct = df.groupby('cat')[f't_{threshold}'].mean()
        for cat_name in cat_names:
            logging_dict[f'val/correct/{cat_name}/Threshold {threshold}'] = cat_correct[cat_name]

    wandb.log(logging_dict)

def wandb_log_train(br, lr):
    df = pd.DataFrame.from_dict({
        'predicted': br.prediction,
        'measured': br.measured,
        'inst_lens': br.inst_lens,
    })

    df['mape'] = abs(df.predicted - df.measured) * 100 / (df.measured + 1e-5)

    logging_dict = {
        "loss/Train": br.loss,
        "lr": lr,
    }

    thresholds = [25, 20, 15, 10, 5, 3]
    for threshold in thresholds:
        logging_dict[f'train/correct/Threshold {threshold}'] = (df.mape < threshold).mean()

    wandb.log(logging_dict)
