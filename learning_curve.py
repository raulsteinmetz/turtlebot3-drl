import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def plot_learning_curve(fpath):
    data = pd.read_csv(fpath)

    ma_episodes_25 = data['scores'].rolling(window=25).mean()
    ma_episodes_100 = data['scores'].rolling(window=100).mean()
    
    plt.figure(figsize=(5, 5))
    plt.plot(data['episode'], ma_episodes_25, color='lightblue', alpha=0.6, label='Moving Average (25)')
    plt.plot(data['episode'], ma_episodes_100, color='blue', label='Moving Average (100)')
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.title('Learning Curve (Episodes)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/lidar{args.lidar}/{args.agent}/stage{args.stage}_episodes.pdf')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(data['steps'], ma_episodes_25, color='lightblue', alpha=0.6, label='Moving Average (25)')
    plt.plot(data['steps'], ma_episodes_100, color='blue', label='Moving Average (100)')
    plt.xlabel('Steps')
    plt.ylabel('Scores')
    plt.title('Learning Curve (Episodes)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/lidar{args.lidar}/{args.agent}/stage{args.stage}_steps.pdf')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot agent's training learning curve")
    parser.add_argument('--agent', type=str, default='sac', help='Specify the RL agent (sac, ddpg, td3, sac_x_hybrid, sac_x)')
    parser.add_argument('--stage', type=int, default=1, help='Specify the environment stage: 1, 2, 3, 4')
    parser.add_argument('--lidar', type=int, default=0, help='Specify the number of LIDAR readings: 10, 360')
    args = parser.parse_args()

    fpath = f'best_models/lidar{args.lidar}/{args.agent}/stage{args.stage}/train.csv'
    plot_learning_curve(fpath)
