from multiAgentPPO import visualize_ppo
from ppo_multiprocessing import visualize_ppo_mp

if __name__ == "__main__":
    visualize_ppo_mp('mp_4policy_ep250', 'mp_4critic_ep250')
