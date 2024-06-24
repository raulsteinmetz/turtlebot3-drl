import argparse
import pandas as pd
from matplotlib import pyplot as plt

def plot_learning_curve(agents, stage, lidar):
    colors = {'ddpg': 'blue', 'td3': 'green', 'sac': 'red', 'dreamer': 'purple'}
    _lidar = lidar
    plt.figure(figsize=(5, 5))
    for agent in agents:
        if int(_lidar) == 0:
            lidar = 360 if agent == 'dreamer' else 10
        fpath = f'best_models/lidar{lidar}/{agent}/stage{stage}/train.csv'
        data = pd.read_csv(fpath)
        
        # for the sake of visualisation
        data['scores'][:10] = 0.0
        data['scores'] = data['scores'].apply(lambda x: 0 if x == -10 else (1 if x == 100 else x))
        
        if stage == 1:
            window = 100
        else: 
            window = 500
        ma_episodes = data['scores'][:600 if stage == 1 and int(_lidar) != 360  else 5000].rolling(window=window, min_periods=1).mean()
        
        std_factor = 0.2
        name = agent if agent != 'dreamer' else 'dreamerv3 (ours)'
        
        plt.plot(data['episode'][:600 if stage == 1 and int(_lidar) != 360 else 5000], ma_episodes, color=colors[agent], label=f'{name}', linewidth=2.0)
    
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title(f'Stage {stage}')
    plt.legend()
    plt.grid(True)
    if int(_lidar) == 0:
        plt.savefig(f'plots/any_lidar/comparison_stage{stage}_episodes.pdf')
    else:
        plt.savefig(f'plots/lidar{lidar}/comparison_stage{stage}_episodes.pdf')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot multiple RL agents' training learning curves")
    parser.add_argument('--stage', type=int, default=1, help='Specify the environment stage: 1, 2, 3, 4')
    parser.add_argument('--lidar', type=int, default=-1, help='Specify the lidar readings: 10 or 360, 0 in for dreamer 360 others 10 comparisson')
    args = parser.parse_args()

    agents = ['ddpg', 'sac', 'td3', 'dreamer']
    plot_learning_curve(agents, args.stage, args.lidar)
