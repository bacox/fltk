from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_fed_time_per_round(df: pd.DataFrame):
    plt.figure()
    sns.boxplot(x='round_id', y='round_duration', data=df, hue='algorithm')
    plt.title('Round duration')
    plt.show()

def plot_fed_acc_per_round(df: pd.DataFrame):
    plt.figure()
    sns.lineplot(x='round_id', y='test_accuracy', data=df, hue='algorithm')
    plt.title('Accuracy over number of rounds')
    plt.show()

def plot_fed_acc_per_time(df: pd.DataFrame):
    plt.figure()
    sns.lineplot(x='timestamp_rel', y='test_accuracy', data=df, hue='algorithm')
    plt.title('Accuracy over time')
    plt.show()

    df['timestamp_rel'] = df['timestamp'].diff()
    print('j')

def load_replication(path: Path):
    replication_id = path.name.split('_')[-1]
    fed_path = path / 'federator.csv'
    print(f'Loading {fed_path}')
    fed_df = pd.read_csv(fed_path)
    print(path, replication_id)
    fed_df['replication'] = replication_id
    fed_df['algorithm'] = path.name.split('_')[-2]
    fed_df['timestamp_rel'] = fed_df['timestamp'].diff()
    return fed_df.fillna(0)


if __name__ == '__main__':
    # path = Path('output/exp_a1_c6')
    # path = Path('output/example_offloading_docker_4')
    path = Path('output/exp_a2_equal')
    fed_dfs = [load_replication(x) for x in path.iterdir()]
    # for subpath in path.iterdir():
    #     load_replication(subpath)
    #     # print(subpath)
    fed_df = pd.concat(fed_dfs, ignore_index=True)
    plot_fed_time_per_round(fed_df)
    plot_fed_acc_per_time(fed_df)
    plot_fed_acc_per_round(fed_df)
    print('Hell')