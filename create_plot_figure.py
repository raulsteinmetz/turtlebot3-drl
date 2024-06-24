import argparse
import pandas as pd
from matplotlib import pyplot as plt

def plot_learning_curves(agents, stages, lidar):
    colors = {'ddpg': 'red', 'td3': 'blue', 'sac': 'gray', 'dreamer': 'darkgreen'} 
    _lidar = lidar
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    axs = axs.flatten()
    
    for i, stage in enumerate(stages):
        ax = axs[i]
        for agent in agents:
            if int(_lidar) == 0:
                lidar = 360 if agent == 'dreamer' else 10
            fpath = f'best_models/lidar{lidar}/{agent}/stage{stage}/train.csv'
            data = pd.read_csv(fpath)
            
            # for the sake of visualisation
            data['scores'][:10] = 0.0
            data['scores'] = data['scores'].apply(lambda x: 0 if x == -10 else (1 if x == 100 else x))
            
            if stage == 1 and lidar == 10:
                window = 100
            else: 
                window = 500
            ma_episodes = data['scores'][:600 if stage == 1 and int(_lidar) != 360  else 5000].rolling(window=window, min_periods=1).mean()

            name = agent if agent != 'dreamer' else 'TurtleDreamer (ours)'
            
            ax.plot(data['episode'][:600 if stage == 1 and int(_lidar) != 360 else 5000], ma_episodes, color=colors[agent], label=f'{name}', linewidth=3.0)
        
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Rewards')
        ax.set_title(f'Stage {stage}')
        ax.legend()
        ax.grid(False)
    
    plt.tight_layout()
    if int(_lidar) == 0:
        plt.savefig(f'plots/any_lidar/comparison_stages_episodes.pdf')
    else:
        plt.savefig(f'plots/lidar{lidar}/comparison_stages_episodes.pdf')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot multiple RL agents' training learning curves")
    parser.add_argument('--stages', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6], help='Specify the environment stages: e.g., 1 2 3 4')
    parser.add_argument('--lidar', type=int, default=-1, help='Specify the lidar readings: 10 or 360, 0 for dreamer 360 others 10 comparison')
    args = parser.parse_args()

    
    agents = ['ddpg', 'sac', 'td3', 'dreamer']
    plot_learning_curves(agents, args.stages, args.lidar)
