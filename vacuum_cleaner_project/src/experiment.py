import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environment import Environment
from simple_agent import SimpleReactiveAgent
from model_agent import ModelBasedAgent

def run_episode(agent_class, env, metric='A', max_steps=500):
    """
    Executa um episódio completo.
    metric: 'A' ignora penalidade de movimento, 'B' considera penalidade.
    Retorna a pontuação total.
    """
    env.reset()
    agent = agent_class()
    total_reward = 0
    steps = 0
    done = False

    while not done and steps < max_steps:
        percept = env.get_percept()
        action = agent.choose_action(percept)
        new_percept, reward, done = env.execute_action(action)

        if agent_class == ModelBasedAgent:
            move_success = False
            if action == 'move_forward':
                move_success = (reward == -1)
            agent.update_after_action(percept, action, move_success)

        total_reward += reward if metric == 'B' else max(reward, 0)  # métrica A: ignora -1
        steps += 1

    return total_reward

def run_experiments(agent_classes, agent_names, env, num_episodes=100, metrics=['A','B']):
    results = {name: {metric: [] for metric in metrics} for name in agent_names}
    for metric in metrics:
        for agent_class, agent_name in zip(agent_classes, agent_names):
            scores = []
            for _ in range(num_episodes):
                score = run_episode(agent_class, env, metric)
                scores.append(score)
            results[agent_name][metric] = scores
    return results

def save_and_plot_results(results, agent_names, metrics, output_dir='results'):
    import os
    os.makedirs(output_dir, exist_ok=True)

    for metric in metrics:
        data = {}
        for name in agent_names:
            data[name] = results[name][metric]
        df = pd.DataFrame(data)
        df.to_csv(f"{output_dir}/scores_metric_{metric}.csv", index=False)

    for metric in metrics:
        means = [np.mean(results[name][metric]) for name in agent_names]
        stds = [np.std(results[name][metric]) for name in agent_names]
        plt.figure(figsize=(8,5))
        plt.bar(agent_names, means, yerr=stds, capsize=10, color=['skyblue', 'salmon'])
        plt.ylabel('Pontuação média')
        plt.title(f'Métrica {metric}')
        plt.savefig(f"{output_dir}/barplot_metric_{metric}.png")
        plt.close()

    for metric in metrics:
        plt.figure(figsize=(8,5))
        data_to_plot = [results[name][metric] for name in agent_names]
        plt.boxplot(data_to_plot, tick_labels=agent_names)
        plt.ylabel('Pontuação')
        plt.title(f'Distribuição das pontuações - Métrica {metric}')
        plt.savefig(f"{output_dir}/boxplot_metric_{metric}.png")
        plt.close()

def main():
    WIDTH, HEIGHT = 6, 6
    DIRT_PROB = 0.3
    OBST_PROB = 0.1
    EPISODES = 100

    env = Environment(WIDTH, HEIGHT, DIRT_PROB, OBST_PROB)
    agent_classes = [SimpleReactiveAgent, ModelBasedAgent]
    agent_names = ["Reativo Simples", "Baseado em Modelo"]

    print("Executando experimentos...")
    results = run_experiments(agent_classes, agent_names, env, EPISODES, metrics=['A','B'])

    for name in agent_names:
        print(f"\n{name}:")
        for metric in ['A','B']:
            scores = results[name][metric]
            print(f"  Métrica {metric}: média = {np.mean(scores):.2f} ± {np.std(scores):.2f}")

    save_and_plot_results(results, agent_names, ['A','B'])
    print("\nResultados salvos na pasta 'results/'")

if __name__ == "__main__":
    main()