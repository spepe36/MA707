import torch
from purrsuit_env import NeuralDashAi
from multiAgentPPO import AgentPolicy, ValueNetwork, flatten_agent_obs
import time


def visualize_ppo_mp(policy_model, critic_model):

    policy = AgentPolicy(9, 5)
    critic = ValueNetwork(9)
    policy.load_state_dict(torch.load(f"models/{policy_model}.pth"))
    critic.load_state_dict(torch.load(f"models/{critic_model}.pth"))
    policy.eval()  # set to evaluation mode

    env = NeuralDashAi(player_speed=2, render_mode='human', number_of_enemies=3)
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

        next_state, reward, truncation, termination, info = env.step(actions, player_ai=False)
        done = {agent: truncation[agent] or termination[agent] for agent in env.agents}
        env.render()
        time.sleep(.005)
        state = next_state


if __name__ == "__main__":
    visualize_ppo_mp('p_8000_sims_3agents', 'c_8000_sims_3agents')