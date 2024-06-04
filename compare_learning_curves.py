import argparse
import pandas as pd
from matplotlib import pyplot as plt

def plot_learning_curve(agents, stage, lidar):
    colors = {'ddpg': 'blue', 'td3': 'green', 'sac': 'red', 'dreamer': 'purple'}

    plt.figure(figsize=(5, 5))
    for agent in agents:
        fpath = f'best_models/lidar{lidar}/{agent}/stage{stage}/train.csv'
        data = pd.read_csv(fpath)
        data['scores'][:10] = 0.0
        
        ma_episodes_10 = data['scores'][:1000 if stage == 1 else 5000].rolling(window=25, min_periods=1).mean()
        ma_episodes_100 = data['scores'][:1000 if stage == 1 else 5000].rolling(window=250, min_periods=1).mean()

        # plt.plot(data['episode'][:1000 if stage == 1 else 5000], ma_episodes_10, color=colors[agent], alpha=0.1, linewidth=2.0)
        name = agent if agent != 'dreamer' else 'dreamerv3 (ours)'
        plt.plot(data['episode'][:1000 if stage == 1 else 5000], ma_episodes_100, color=colors[agent], label=f'{name}')
    
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title(f'Reward Moving Average (n = 250) on Stage {stage}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/lidar{lidar}/comparison_stage{stage}_episodes.pdf')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot multiple RL agents' training learning curves")
    parser.add_argument('--stage', type=int, default=1, help='Specify the environment stage: 1, 2, 3, 4')
    parser.add_argument('--lidar', type=int, default=1, help='Specify the lidar readings: 10 or 360')
    args = parser.parse_args()

    agents = ['ddpg', 'sac', 'td3', 'dreamer']
    plot_learning_curve(agents, args.stage, args.lidar)
