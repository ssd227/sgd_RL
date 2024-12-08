
import matplotlib.pyplot as plt


# show rewards image for log
def show_rewards(episode, episode_rewards):
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    # plt.show()
    plt.savefig('rewards'+str(episode)+'.png', dpi=300, bbox_inches="tight")
    plt.clf()
    