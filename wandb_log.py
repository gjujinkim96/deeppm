import wandb
import pandas as pd
from utils import recursive_vars

def wandb_init(args, cfg):
    mode = 'disabled' if args.wandb_disabled else 'online'

    tags = ['v4:data_split']

    config = recursive_vars(cfg)
    config['small_size'] = args.small_size
    config['small_training'] = args.small_training
    wandb.init(
        project='deeppm',
        config=config,
        mode=mode,
        name=args.exp_name,
        tags=tags,
    )

    wandb.run.log_code(include_fn=lambda path: path.endswith(".py") or \
                path.endswith(".sh") or path.endswith(".json") or path.endswith(".yaml"))


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

best_val_loss = float('inf')
best_mape_loss = float('inf')
best_25_accuracy = float('-inf')

best_scatter_data = []
best_threshold_data = []
best_inst_len_mean_data = []
best_inst_len_correct_data = []

def wandb_log_val(er, epoch):
    df = pd.DataFrame.from_dict({
        'predicted': er.prediction,
        'measured': er.measured,
        'inst_lens': er.inst_lens,
        'index': er.index,
    })

    df['mape'] = abs(df.predicted - df.measured) * 100 / (df.measured + 1e-5)
    df['correct'] = df.mape < 25
    inst_len_mean = df.groupby('inst_lens').mape.mean()
    inst_len_correct = df.groupby('inst_lens').correct.mean()

    logging_dict = {
        "loss/Validation": er.loss,
        "mape/Validation": er.mape,
        'epoch': epoch,
    }

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

    thresholds = [25]
    for threshold in thresholds:
        df[f't_{threshold}'] = df.mape < threshold
        cat_correct = df.groupby('cat')[f't_{threshold}'].mean()
        for cat_name in cat_names:
            logging_dict[f'val/correct/threshold_{threshold}/{cat_name}'] = cat_correct[cat_name]
    
    df['t_25'] = df.mape < 25
    cat_correct = df.groupby('cat')['t_25'].mean()
    for cat_name in cat_names:
        logging_dict[f'val/correct/threshold_25/{cat_name}'] = cat_correct[cat_name]


    global best_val_loss
    global best_mape_loss
    global best_25_accuracy

    global best_scatter_data
    global best_threshold_data
    global best_inst_len_mean_data
    global best_inst_len_correct_data

    if er.mape < best_mape_loss:
        best_mape_loss = er.mape
        wandb.run.summary['best_mape'] = best_mape_loss

    if er.loss < best_val_loss:
        best_val_loss = er.loss
        wandb.run.summary['best_loss'] = best_val_loss
    
    scatter_data = [[x, y, z, a] for (x, y, z, a) in zip(df.predicted, df.measured, df.inst_lens, df.index)]
    threshold_data = [[i, (df.mape < i).mean()] for i in range(1, 100)]
    inst_len_mean_data = [[0, 0]] + [[inst_len, mape] for inst_len, mape in inst_len_mean.items()] + [[max(inst_len_mean.index) + 1, 0]]
    inst_len_correct_data = [[0, 0]] + [[inst_len, correct] for inst_len, correct in inst_len_correct.items()] + [[max(inst_len_mean.index) + 1, 0]]

    cur_accuracy = (df.mape < 25).mean().item()
    if  cur_accuracy > best_25_accuracy:
        best_25_accuracy = cur_accuracy
        wandb.run.summary['best_accr_25'] = best_25_accuracy

        best_scatter_data = scatter_data
        best_threshold_data = threshold_data
        best_inst_len_mean_data = inst_len_mean_data
        best_inst_len_correct_data = inst_len_correct_data

    # best
    scatter_table = wandb.Table(data=best_scatter_data, columns = ["Predicted", "Measured", "Inst_Len", "Index"])
    logging_dict["val/summary/best_scatter"] = wandb.plot.scatter(scatter_table, "Predicted", "Measured", title="Best Measured vs. Predicted")

    threshold_table = wandb.Table(data=best_threshold_data, columns=["Threshold", "Correct"])
    logging_dict["val/summary/best_threshold"] = wandb.plot.line(threshold_table, "Threshold", "Correct", title="Best Threshold vs. Correct")

    threshold_table = wandb.Table(data=best_inst_len_mean_data, columns=["Inst_Len", "Mape"])
    logging_dict["val/summary/best_inst_mape"] = wandb.plot.line(threshold_table, "Inst_Len", "Mape", title="Best Mape grouped by Inst len")
    
    threshold_table = wandb.Table(data=best_inst_len_correct_data, columns=["Inst_Len", "Correct"])
    logging_dict["val/summary/best_inst_correct"] = wandb.plot.line(threshold_table, "Inst_Len", "Correct", title="Best Correct % grouped by Inst len")

    #  cur
    scatter_table = wandb.Table(data=scatter_data, columns = ["Predicted", "Measured", "Inst_Len", "Index"])
    logging_dict["val/summary/scatter"] = wandb.plot.scatter(scatter_table, "Predicted", "Measured", title="Measured vs. Predicted")

    threshold_table = wandb.Table(data=threshold_data, columns=["Threshold", "Correct"])
    logging_dict["val/summary/threshold"] = wandb.plot.line(threshold_table, "Threshold", "Correct", title="Threshold vs. Correct")

    threshold_table = wandb.Table(data=inst_len_mean_data, columns=["Inst_Len", "Mape"])
    logging_dict["val/summary/inst_mape"] = wandb.plot.line(threshold_table, "Inst_Len", "Mape", title="Mape grouped by Inst len")
    
    threshold_table = wandb.Table(data=inst_len_correct_data, columns=["Inst_Len", "Correct"])
    logging_dict["val/summary/inst_correct"] = wandb.plot.line(threshold_table, "Inst_Len", "Correct", title="Correct % grouped by Inst len")

    wandb.log(logging_dict)

def wandb_log_test(er):
    df = pd.DataFrame.from_dict({
        'predicted': er.prediction,
        'measured': er.measured,
        'inst_lens': er.inst_lens,
    })

    df['mape'] = abs(df.predicted - df.measured) * 100 / (df.measured + 1e-5)

    logging_dict = {
    }

    wandb.run.summary['test/loss'] = er.loss
    wandb.run.summary['test/mape'] = er.mape

    
    scatter_data = [[x, y] for (x, y) in zip(df.predicted, df.measured)]
    
    scatter_table = wandb.Table(data=scatter_data, columns = ["Predicted", "Measured"])
    logging_dict["test/summary/scatter"] = wandb.plot.scatter(scatter_table, "Predicted", "Measured", title="Measured vs. Predicted")

    threshold_data = [[i, (df.mape < i).mean()] for i in range(1, 100)]
    threshold_table = wandb.Table(data=threshold_data, columns=["Threshold", "Correct"])
    logging_dict["test/summary/threshold"] = wandb.plot.line(threshold_table, "Threshold", "Correct", title="Threshold vs. Correct")


    thresholds = [25, 20, 15, 10, 5, 3]
    for threshold in thresholds:
        wandb.run.summary[f'test/correct/Threshold {threshold}'] = (df.mape < threshold).mean()


    df['cat'] = pd.Categorical(df.inst_lens.apply(cat_by_inst), ordered=True, categories=cat_names)
    cat_mape_mean = df.groupby('cat').mape.mean()
    for cat_name in cat_names:
        wandb.run.summary[f'test/cat/MAPE Error {cat_name}'] = cat_mape_mean[cat_name]

    mape_data = [[label, val] for (label, val) in zip(cat_names, cat_mape_mean)]
    mape_table = wandb.Table(data=mape_data, columns = ["Number of Instructions", "Mean MAPE Error"])
    logging_dict["test/summary/mean error"] = wandb.plot.bar(mape_table, "Number of Instructions", 'Mean MAPE Error', title="Mean Error Over Size")

    df['t_25'] = df.mape < threshold
    cat_correct = df.groupby('cat')['t_25'].mean()
    for cat_name in cat_names:
        wandb.run.summary[f'test/correct/threshold_25/{cat_name}'] = cat_correct[cat_name]

    wandb.log(logging_dict)

def wandb_log_train(br, lr, epoch):
    df = pd.DataFrame.from_dict({
        'predicted': br.prediction,
        'measured': br.measured,
        'inst_lens': br.inst_lens,
        # 'index': br.index,
    })

    df['mape'] = abs(df.predicted - df.measured) * 100 / (df.measured + 1e-5)

    logging_dict = {
        "loss/Train": br.loss,
        "mape/Train": br.mape,
        "lr": lr,
        'epoch': epoch,
    }

    thresholds = [25, 20, 15, 10, 5, 3]
    for threshold in thresholds:
        logging_dict[f'train/correct/Threshold {threshold}'] = (df.mape < threshold).mean()

    wandb.log(logging_dict)
