import wandb
import pandas as pd
from utils import recursive_vars

def make_cat_names(cat_limits):
    ret = []
    c = 'A'
    start = 1
    for cat_limit in cat_limits:
        ret.append(f'{c}:{start}~{cat_limit-1}')
        start = cat_limit
        c = chr(ord(c) + 1)
    ret.append(f'{c}:{start}~')
    return ret


cat_limits = [23, 50, 100, 150, 200]
cat_names = make_cat_names(cat_limits)

class CategoryNames:
    def __init__(self, cat_limits):
        self.limits = cat_limits
        self.names = make_cat_names(self.limits)
    
    def idx_map(self, inst_len):
        for idx, cat_limit in enumerate(self.limits):
            if inst_len < cat_limit:
                return idx
        return idx + 1

    def name_map(self, inst_len):
        return self.names[self.idx_map(inst_len)]

cat_limits = [23, 50, 100, 150, 200]
CATS = CategoryNames(cat_limits)

THRESHOLDS = [25, 20, 15, 10, 5, 3]

def wandb_init(args, cfg, group=None):
    wandb_cfg = cfg.log.wandb
    mode = getattr(wandb_cfg, 'mode', 'online')
    mode = 'disabled' if args.wandb_disabled else 'online'

    tags = getattr(wandb_cfg, 'tags', [])

    config = recursive_vars(cfg)
    config['small_size'] = args.small_size
    config['small_training'] = args.small_training
    wandb.init(
        project=getattr(wandb_cfg, 'project', None),
        config=config,
        mode=mode,
        name=args.exp_name,
        group=group,
        tags=tags,
    )

    wandb.run.log_code(include_fn=lambda path: path.endswith(".py") or \
                path.endswith(".sh") or path.endswith(".json") or path.endswith(".yaml"))

def wandb_test_init(args, cfg, model_epoch, date):
    wandb_cfg = cfg.log.wandb
    mode = getattr(wandb_cfg, 'mode', 'online')
    mode = 'disabled' if args.wandb_disabled else 'online'

    if args.resume_id is None:
        tags = getattr(wandb_cfg, 'tags', [])
        tags.append('test')

        config = recursive_vars(cfg)
        config['small_size'] = args.small_size
        config['exp/name'] = args.exp_name
        config['exp/date'] = date
        config['exp/model_type'] = args.type
        config['exp/model_epoch'] = model_epoch

        wandb.init(
            project=getattr(wandb_cfg, 'project', None),
            config=config,
            mode=mode,
            name=f'{args.exp_name}/{date}/{args.type}/{model_epoch}',
            tags=tags,
        )

        wandb.run.log_code(include_fn=lambda path: path.endswith(".py") or \
                    path.endswith(".sh") or path.endswith(".json") or path.endswith(".yaml"))
    else:
        wandb.init(cfg.log.wandb.project, id=args.resume_id, resume='must', mode=mode)
    
def wandb_finish():
    wandb.finish()


def make_df_from_batch_result(result):
    df = pd.DataFrame.from_dict({
        'predicted': result.prediction,
        'measured': result.measured,
        'inst_lens': result.inst_lens,
        'index': result.index,
    })

    df['mape'] = abs(df.predicted - df.measured) * 100 / (df.measured + 1e-5)
    df['correct'] = df.mape < 25
    df['cat'] = pd.Categorical(df.inst_lens.apply(CATS.name_map), ordered=True, categories=CATS.names)
    return df


def make_tile_and_type_prefix(log_prefix):
    if len(log_prefix) == 0:
        return '', ''
    return log_prefix.capitalize() + ' ', log_prefix.lower() + '_'

def log_scatter(logging_dict, df, log_type, log_prefix=''):
    title_prefix, type_prefix = make_tile_and_type_prefix(log_prefix)

    scatter_data = [[x, y, z, a] for (x, y, z, a) in zip(df.predicted, df.measured, df.inst_lens, df.index)]
    scatter_table = wandb.Table(data=scatter_data, columns = ["Predicted", "Measured", "Inst_Len", "Index"])
    logging_dict[f'{log_type}/summary/{type_prefix}scatter'] = wandb.plot.scatter(scatter_table, "Predicted", "Measured", title=f'{title_prefix}Measured vs. Predicted')

def log_threshold(logging_dict, df, log_type, log_prefix=''):
    title_prefix, type_prefix = make_tile_and_type_prefix(log_prefix)

    threshold_data = [[i, (df.mape < i).mean()] for i in range(1, 100)]
    threshold_table = wandb.Table(data=threshold_data, columns=["Threshold", "Correct"])
    logging_dict[f'{log_type}/summary/{type_prefix}threshold'] = wandb.plot.line(threshold_table, "Threshold", "Correct", title=f'{title_prefix}Threshold vs. Correct')

def log_inst_mape(logging_dict, df, log_type, log_prefix=''):
    title_prefix, type_prefix = make_tile_and_type_prefix(log_prefix)
    inst_len_mean = df.groupby('inst_lens', observed=False).mape.mean()

    inst_len_mean_data = [[0, 0]] + [[inst_len, mape] for inst_len, mape in inst_len_mean.items()] + [[max(inst_len_mean.index) + 1, 0]]
    inst_len_mean_table = wandb.Table(data=inst_len_mean_data, columns=["Inst_Len", "Mape"])
    logging_dict[f'{log_type}/summary/{type_prefix}inst_mape'] = wandb.plot.line(inst_len_mean_table, "Inst_Len", "Mape", title=f'{title_prefix}Mape grouped by Inst len')

def log_inst_correct(logging_dict, df, log_type, log_prefix=''):
    title_prefix, type_prefix = make_tile_and_type_prefix(log_prefix)
    inst_len_correct = df.groupby('inst_lens', observed=False).correct.mean()

    inst_len_correct_data = [[0, 0]] + [[inst_len, correct] for inst_len, correct in inst_len_correct.items()] + [[max(inst_len_correct.index) + 1, 0]]
    inst_len_correct_table = wandb.Table(data=inst_len_correct_data, columns=["Inst_Len", "Correct"])
    logging_dict[f'{log_type}/summary/{type_prefix}inst_correct'] = wandb.plot.line(inst_len_correct_table, "Inst_Len", "Correct", title=f'{title_prefix}Correct % grouped by Inst len')

def log_cat_mean_error(logging_dict, df, log_type, log_prefix=''):
    title_prefix, type_prefix = make_tile_and_type_prefix(log_prefix)

    cat_mape_mean = df.groupby('cat', observed=False).mape.mean()
    mape_data = [[label, val] for (label, val) in zip(CATS.names, cat_mape_mean)]
    mape_table = wandb.Table(data=mape_data, columns = ["Number of Instructions", "Mean MAPE Error"])
    logging_dict[f'{log_type}/summary/{type_prefix}mean error'] = wandb.plot.bar(mape_table, "Number of Instructions", 'Mean MAPE Error', title=f'{title_prefix}Mean Error Over Size')

def log_correct_threshold(logging_dict, df, log_type, log_prefix=''):
    title_prefix, type_prefix = make_tile_and_type_prefix(log_prefix)
    for threshold in THRESHOLDS:
        logging_dict[f'{log_type}/correct/{type_prefix}Threshold {threshold}'] = (df.mape < threshold).mean()

def log_cat_mape(logging_dict, df, log_type, log_prefix=''):
    title_prefix, type_prefix = make_tile_and_type_prefix(log_prefix)
    cat_mape_mean = df.groupby('cat', observed=False).mape.mean()
    for cat_name in CATS.names:
        logging_dict[f'{log_type}/cat/{type_prefix}MAPE Error {cat_name}'] = cat_mape_mean[cat_name]

def log_correct_threshold_25_cat(logging_dict, df, log_type, log_prefix=''):
    title_prefix, type_prefix = make_tile_and_type_prefix(log_prefix)

    cat_correct = df.groupby('cat', observed=False).correct.mean()
    for cat_name in CATS.names:
        logging_dict[f'{log_type}/correct/threshold_25/{type_prefix}{cat_name}'] = cat_correct[cat_name]

best_val_25_accuracy = float('-inf')
def wandb_log_val(er, epoch):
    df = make_df_from_batch_result(er)
    er_mape = df.mape.mean().item()

    logging_dict = {
        "loss/Validation": er.loss,
        "mape/Validation": er_mape,
        'epoch': epoch,
    }

    log_cat_mean_error(logging_dict, df, 'val')
    log_cat_mape(logging_dict, df, 'val')
    log_correct_threshold(logging_dict, df, 'val')
    log_correct_threshold_25_cat(logging_dict, df, 'val')

    global best_val_25_accuracy
    
    # best
    cur_accuracy = (df.mape < 25).mean().item()
    if  cur_accuracy >= best_val_25_accuracy:
        best_val_25_accuracy = cur_accuracy
        wandb.run.summary['best_mape'] = er_mape
        wandb.run.summary['best_loss'] = er.loss
        wandb.run.summary['best_accr_25'] = best_val_25_accuracy

        log_scatter(logging_dict, df, 'val', 'best')
        log_threshold(logging_dict, df, 'val', 'best')
        log_inst_mape(logging_dict, df, 'val', 'best')
        log_inst_correct(logging_dict, df, 'val', 'best')

        log_cat_mean_error(logging_dict, df, 'val', 'best')

    wandb.log(logging_dict)

def wandb_log_test(er):
    df = make_df_from_batch_result(er)
    er_mape = df.mape.mean().item()

    logging_dict = {
    }

    wandb.run.summary['test/loss'] = er.loss
    wandb.run.summary['test/mape'] = er_mape


    log_cat_mean_error(logging_dict, df, 'test')
    log_cat_mape(wandb.run.summary, df, 'test')
    log_correct_threshold(wandb.run.summary, df, 'test')
    log_correct_threshold_25_cat(wandb.run.summary, df, 'test')

    
    log_scatter(logging_dict, df, 'test')
    log_threshold(logging_dict, df, 'test')
    log_inst_mape(logging_dict, df, 'test')
    log_inst_correct(logging_dict, df, 'test')
    
    wandb.log(logging_dict)
    
def wandb_log_train(br, lr, epoch):
    df = make_df_from_batch_result(br)
    br_mape = df.mape.mean().item()

    logging_dict = {
        "loss/Train": br.loss,
        "mape/Train": br_mape,
        "lr": lr,
        'epoch': epoch,
    }

    log_correct_threshold(logging_dict, df, 'train')

    wandb.log(logging_dict)
