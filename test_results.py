import argparse
import pandas as pd

def print_test_results(agents, stage):
    for agent in agents:
        fpath = f'best_models/{agent}/stage{stage}/test.csv'
        data = pd.read_csv(fpath)
        
        # Calculate the percentage of scores that are exactly 100
        total_scores = len(data['reward'])
        scores_100 = sum(data['reward'] == 100)
        percentage_100 = (scores_100 / total_scores) * 100 if total_scores > 0 else 0
        
        # Print the results
        print(f"{agent} - Percentage of scores equal to 100: {percentage_100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Print test results for each algorithm")
    parser.add_argument('--stage', type=int, default=1, help='Specify the environment stage: 1, 2, 3, 4')
    args = parser.parse_args()

    agents = ['ddpg', 'td3', 'sac', 'dreamer']  # Make sure 'dreamer' is a valid agent name and is consistent with your dataset
    print_test_results(agents, args.stage)
