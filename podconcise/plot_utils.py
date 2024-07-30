from typing import Sequence, Dict
from matplotlib import pyplot as plt


def plot_learning_curve(log_history:Sequence[Dict], eval_metric:str=None):
    """
    Given a training result log history, plot the train and validation losses and optionally
    the validation metric.
    /!\ take the assumption that the results saving strategy is steps.
    """
    train_loss_by_step = {}
    eval_loss_by_step = {}
    eval_metric_by_step = {}

    for log in log_history:
        step = log.get("step", None)
        if step is None:
            continue
        train_loss = log.get("loss", None)
        if train_loss:
            train_loss_by_step[step] = train_loss
        eval_loss = log.get("eval_loss", None)
        if eval_loss:
            eval_loss_by_step[step] = eval_loss
        eval_f1_plus_log_likelihood = log.get(eval_metric, None)
        if eval_f1_plus_log_likelihood:
            eval_metric_by_step[step] = eval_f1_plus_log_likelihood

    steps, train_loss = zip(*[(k,v) for k,v in train_loss_by_step.items()])
    plt.plot(steps, train_loss, label="train_loss", linestyle=":")
    plt.scatter(steps, train_loss)

    steps, eval_loss = zip(*[(k,v) for k,v in eval_loss_by_step.items()])
    plt.plot(steps, eval_loss, label='eval_loss', linestyle=":")
    plt.scatter(steps, eval_loss)

    if len(eval_metric_by_step) > 0:
        steps, eval_metric = zip(*[(k,v) for k,v in eval_metric_by_step.items()])
        plt.plot(steps, eval_metric, label='eval_metric', linestyle=":")
        plt.scatter(steps, eval_metric)


    ax = plt.gca()
    ax.set_xlabel('steps')
    ax.legend();