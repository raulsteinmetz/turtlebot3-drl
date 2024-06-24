import argparse
import pandas as pd
from matplotlib import pyplot as plt

def plot_learning_curves(agents, stages, lidar):
    colors = {'ddpg': 'orange', 'td3': 'blue', 'sac': 'red', 'dreamer': 'green'} 
    _lidar = lidar
    
    # First figure with 3x2 layout
    fig1, axs1 = plt.subplots(3, 2, figsize=(10, 15))  # 3 rows, 2 columns
    axs1 = axs1.flatten()
    
    # Second figure with 2x3 layout
    fig2, axs2 = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns
    axs2 = axs2.flatten()
    
    for i, stage in enumerate(stages):
        ax1 = axs1[i]
        ax2 = axs2[i]
        
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
            ma_episodes = data['scores'][:600 if stage == 1 and int(_lidar) != 360 else 5000].rolling(window=window, min_periods=1).mean()

            name = agent if agent != 'dreamer' else 'TurtleDreamer (ours)'
            
            ax1.plot(data['episode'][:600 if stage == 1 and int(_lidar) != 360 else 5000], ma_episodes, color=colors[agent], label=f'{name}', linewidth=3.0, alpha=0.4 if agent != 'dreamer' else 1)
            ax2.plot(data['episode'][:600 if stage == 1 and int(_lidar) != 360 else 5000], ma_episodes, color=colors[agent], label=f'{name}', linewidth=3.0, alpha=0.4 if agent != 'dreamer' else 1)
        
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Rewards')
        ax1.set_title(f'Stage {stage}')
        ax1.legend()
        ax1.grid(False)
        
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Rewards')
        ax2.set_title(f'Stage {stage}')
        ax2.legend()
        ax2.grid(False)
    
    plt.tight_layout()
    if int(_lidar) == 0:
        fig1.savefig(f'plots/any_lidar/comparison_stages_episodes_3x2.pdf')
        fig2.savefig(f'plots/any_lidar/comparison_stages_episodes_2x3.pdf')
    else:
        fig1.savefig(f'plots/lidar{lidar}/comparison_stages_episodes_3x2.pdf')
        fig2.savefig(f'plots/lidar{lidar}/comparison_stages_episodes_2x3.pdf')
    plt.close(fig1)
    plt.close(fig2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot multiple RL agents' training learning curves")
    parser.add_argument('--stages', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6], help='Specify the environment stages: e.g., 1 2 3 4')
    parser.add_argument('--lidar', type=int, default=-1, help='Specify the lidar readings: 10 or 360, 0 for dreamer 360 others 10 comparison')
    args = parser.parse_args()

    agents = ['ddpg', 'sac', 'td3', 'dreamer']
    plot_learning_curves(agents, args.stages, args.lidar)
