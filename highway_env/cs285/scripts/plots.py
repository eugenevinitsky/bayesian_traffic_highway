import os
import matplotlib.pyplot as plt

import tensorflow as tf

# need to run script at correct location
DATA_DIR =  os.getcwd() + "/data/"

def get_section_results(file, v_tag):
    eval_returns = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == v_tag:
                eval_returns.append(v.simple_value)

    return eval_returns

def get_x_y_std(env_name, training_type, data_len=10, expert_y=False):
    """Given the name of the data type e.g. env_name, BC_ANT_1, DAGGER_ANT, BC_HOPPER_1,

    return the appropriate x, y, std lists"""

    data_dir_nodes = os.listdir(DATA_DIR)
    # import ipdb; ipdb.set_trace()
    for f in data_dir_nodes:
        if env_name in f and training_type in f:
            path = DATA_DIR + f + '/'           


    # for DAGGER, BC_*_1, there'll be only one tf_path
    # also, don't need the hyperparam tuning guy this time
    for tf_path in os.listdir(path):
        full_tf_path = path + tf_path
        if expert_y:
            y = get_section_results(full_tf_path, "Initial_DataCollection_AverageReturn")
        else:
            y = get_section_results(full_tf_path, "Eval_AverageReturn")
        std = get_section_results(full_tf_path, "Eval_StdReturn")
        

    x = [i for i in range(10)]

    # fill in y values if we're plotting a constant BC plot
    if len(y) == 1:
        y = [y[0] for _ in range(10)]
        
    return x, y, std

def plot(x, y, save_name):
    plt.title("num_agent_train_steps_per_iter vs Eval_AverageReturn")
    plt.xlabel("num_agent_train_steps_per_iter")
    plt.ylabel("Eval_AverageReturn")

    plt.plot(x, y)
    plt.savefig(save_name)
    plt.show()

def plot_summary(env_name):
    import numpy as np
    x = np.arange(10)
    fig = plt.figure()
    plt.title(env_name + " Number of DAgger training steps vs Policy's mean training returns")

    plt.xlabel("Training iteration number")    
    plt.ylabel("Return values")
    plt.ylim(0, 6000)

    ax = fig.add_subplot(111)

    x, dagger_y, dagger_std = get_x_y_std(env_name, "dagger", expert_y=False)
    x, expert_y, expert_std = get_x_y_std(env_name, "bc", expert_y=True)
    x, policy_y, expert_std = get_x_y_std(env_name, "bc", expert_y=False)

    ax.plot(x, policy_y, c='b',marker="^", label='Behavior cloning agent mean return')
    ax.plot(x, expert_y, c='g',marker=(8,2,0), label='Expert policy')
    ax.errorbar(x, dagger_y, yerr=dagger_std, c='k', label='Dagger policy mean return')

    plt.savefig(env_name + " DAgger results")
    plt.legend(loc=9)
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, help='choices: ant, hopper', required=True)
    parser.add_argument('--tune_hyperparam', type=bool, default=False)
    args = parser.parse_args()
    params = vars(args)

    env_name = args.env_name
    tune_hyperparam = args.tune_hyperparam

    if not tune_hyperparam:
        plot_summary(env_name)

    else:
        data_dir_nodes = os.listdir(DATA_DIR)

        x, y = [], []

        for f in data_dir_nodes:
            if "q1_bc_ant_20" in f:
                q1_bc_path = DATA_DIR + f + '/'           

        # for DAGGER, BC_*_1, there'll be only one tf_path
        for tf_path in os.listdir(q1_bc_path):
            full_tf_path = q1_bc_path + tf_path
            mean = get_section_results(full_tf_path, "Eval_AverageReturn")[0]
            num_agent_train_steps_per_iter = get_section_results(full_tf_path, "num_agent_train_steps_per_iter")[0]
            x.append(num_agent_train_steps_per_iter)
            y.append(mean)

        lst = list(zip(x, y))
        lst.sort()

        new_x, new_y = [], []

        for left, right in lst:
            new_x.append(left)
            new_y.append(right)


        
        plot(new_x, new_y, 'Ants_tr_steps_vs_mean.png')



if __name__ == "__main__":
    main()
