import torch
from numpy import number

from multiAgentEnv import NeuralDash
from multiAgentPPO import save_model, update_policy, init_weights, AgentPolicy, ValueNetwork, \
    compute_discounted_rewards, compute_gae, flatten_agent_obs
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import time


def collect_rollout(policy_state_dict, critic_state_dict, obs_dims, action_dims, rollout_steps, gamma, lam, render_mode):
    device = torch.device("cpu")
    # Create a separate environment in each worker
    env = NeuralDash(render_mode=render_mode, number_of_enemies=4, env_width=900, env_height=900)
    env.reset()

    # Create local models
    local_policy = AgentPolicy(obs_dims, action_dims).to(device)
    local_critic = ValueNetwork(obs_dims).to(device)

    local_policy.load_state_dict(policy_state_dict)
    local_critic.load_state_dict(critic_state_dict)

    states = env.reset()
    transitions = {agent: [] for agent in env.agents}
    rewards = {agent: [] for agent in env.agents}
    values = {agent: [] for agent in env.agents}
    done = {agent: False for agent in env.agents}

    episode_reward = 0.0  # track total reward for logging (not strictly necessary to return for training)

    for step in range(rollout_steps):
        actions = {}
        obs_tensors = {}
        for agent in env.agents:
            if not done[agent]:
                max_pos = 900
                flattened_state = flatten_agent_obs(states[agent])
                normalized_state = flattened_state / max_pos
                obs_tensor = torch.tensor(normalized_state, dtype=torch.float32, device=device)
                logits = local_policy(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).item()

                actions[agent] = action
                obs_tensors[agent] = obs_tensor

        next_states, rewards_step, truncations, terminations, _ = env.step(actions)
        done = {agent: truncations[agent] or terminations[agent] for agent in truncations.keys()}

        for agent in env.agents:
            if not done[agent]:
                transitions[agent].append({
                    "state": states[agent],
                    "action": actions[agent],
                    "reward": rewards_step[agent],
                    "next_state": next_states[agent],
                    "done": done[agent],
                })
                rewards[agent].append(rewards_step[agent])
                values[agent].append(local_critic(obs_tensors[agent]).item())

            episode_reward += rewards_step[agent]

        states = next_states

        # If all agents are done, break early
        if all(done.values()):
            break

    # Compute advantages and returns
    advantages = {}
    returns = {}
    for agent in env.agents:
        if not done[agent]:
            next_value = local_critic(torch.tensor(flatten_agent_obs(states[agent]), dtype=torch.float32, device=device)).item()
        else:
            next_value = 0.0
        advantages[agent] = compute_gae(rewards[agent], values[agent], next_value, gamma=gamma, lam=lam)
        returns[agent] = compute_discounted_rewards(rewards[agent], gamma=gamma)

    # Flatten out transitions, advantages, and returns
    combined_transitions = []
    combined_advantages = []
    combined_returns = []
    for agent in env.agents:
        combined_transitions.extend(transitions[agent])
        combined_advantages.extend(advantages[agent])
        combined_returns.extend(returns[agent])

    # Return the collected data
    return combined_transitions, combined_advantages, combined_returns, episode_reward


def train_ppo(
    num_episodes=100,
    rollout_steps=2000,
    gamma=0.99,
    lam=0.95,
    render_mode='False',
    policy_path=None,
    critic_path=None,
    num_envs=8,
    name="mp"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a single "dummy" env just to get observation/action dims
    dummy_env = NeuralDash(render_mode=render_mode)
    obs_dims = 9
    action_dims = 5

    policy = AgentPolicy(obs_dims, action_dims).to(device)
    old_policy = AgentPolicy(obs_dims, action_dims).to(device)
    critic = ValueNetwork(obs_dims).to(device)

    # Load saved models if provided
    if policy_path is not None and critic_path is not None:
        policy_state_dict = torch.load(policy_path, map_location=device)
        critic_state_dict = torch.load(critic_path, map_location=device)
        policy.load_state_dict(policy_state_dict)
        critic.load_state_dict(critic_state_dict)
    else:
        # Initialize weights if no model provided
        policy.apply(init_weights)
        critic.apply(init_weights)

    old_policy.load_state_dict(policy.state_dict())

    optimizer_policy = torch.optim.Adam(policy.parameters(), lr=.00005)
    optimizer_value = torch.optim.Adam(critic.parameters(), lr=.00005)

    episode_rewards_log = []

    # Multiprocessing pool
    # Use a Pool with processes = num_envs
    with Pool(processes=num_envs) as pool:
        for episode in range(num_episodes):
            # Send the current policy/critic params to all workers
            # We'll use CPU versions for rollout to avoid GPU overhead across multiple processes
            policy_cpu_state = {k: v.cpu() for k, v in policy.state_dict().items()}
            critic_cpu_state = {k: v.cpu() for k, v in critic.state_dict().items()}

            # Prepare arguments for all processes
            worker_args = [
                (policy_cpu_state, critic_cpu_state, obs_dims, action_dims, rollout_steps, gamma, lam, render_mode)
                for _ in range(num_envs)
            ]

            # Collect rollouts from all parallel environments
            results = pool.starmap(collect_rollout, worker_args)

            # results is a list of tuples (transitions, advantages, returns, episode_reward)
            all_transitions = []
            all_advantages = []
            all_returns = []
            total_reward_this_iteration = 0.0

            for (transitions, advantages, returns, ep_reward) in results:
                all_transitions.extend(transitions)
                all_advantages.extend(advantages)
                all_returns.extend(returns)
                total_reward_this_iteration += ep_reward

            # Perform PPO update with collected data
            # Move data to device if needed
            update_policy(
                policy, old_policy, critic, all_transitions, all_advantages, all_returns,
                optimizer_policy, optimizer_value, device=device
            )

            old_policy.load_state_dict(policy.state_dict())

            avg_reward = total_reward_this_iteration / num_envs
            episode_rewards_log.append(avg_reward)

            if episode == num_episodes // 2:
                save_model(policy, critic, name, directory="models", episode=episode)

            if episode == num_episodes - 1:
                save_model(policy, critic, name, directory="models", episode=episode)

            print(f"Episode {episode + 1}/{num_episodes} completed. Avg Episode Reward (across {num_envs} envs): {avg_reward}")

    # After training, plot reward curve
    plt.figure()
    plt.plot(episode_rewards_log)
    plt.xlabel('Episode')
    plt.ylabel('Average Total Reward (Parallel)')
    plt.title('Training Reward Curve')
    plt.savefig('reward_curve.png')
    plt.close()
    print("Saved reward curve plot as reward_curve.png")

    # Plot action distribution from a sample state
    sample_state = np.zeros(obs_dims, dtype=np.float32)
    with torch.no_grad():
        obs_tensor = torch.tensor(sample_state, dtype=torch.float32, device=device)
        logits = policy(obs_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    plt.figure()
    actions = np.arange(action_dims)
    plt.bar(actions, probs)
    plt.xlabel('Action')
    plt.ylabel('Probability')
    plt.title('Action Distribution at Sample State')
    plt.savefig('action_distribution.png')
    plt.close()
    print("Saved action distribution plot as action_distribution.png")


def visualize_ppo_mp(policy_model, critic_model, obs):

    policy = AgentPolicy(obs, 5)
    critic = ValueNetwork(obs)
    policy.load_state_dict(torch.load(f"models/{policy_model}.pth"))
    critic.load_state_dict(torch.load(f"models/{critic_model}.pth"))
    policy.eval()

    if obs == 9:
        enemies = 3
    else:
        enemies = 4

    env = NeuralDash(number_of_enemies=enemies, render_mode='human')
    state = env.reset()

    done = {agent: False for agent in env.agents}
    while not all(done.values()):
        actions = {}
        for agent in env.agents:
            if not done[agent]:
                obs = torch.tensor(flatten_agent_obs(state[agent]), dtype=torch.float32)
                logits = policy(obs)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).item()
                actions[agent] = action

        next_state, reward, truncation, termination, info = env.step(actions)
        done = {agent: truncation[agent] or termination[agent] for agent in env.agents}
        env.render()
        time.sleep(.005)
        state = next_state


if __name__ == "__main__":
    train_ppo(num_episodes=500, rollout_steps=1500, gamma=0.99, lam=0.95, render_mode='False', num_envs=10, name="mp_4")
    # visualize_ppo_mp('p_8000_sims_3agents', 'c_8000_sims_3agents', 9)


