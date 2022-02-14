# Great Lunar Lander with DQN

Authors: Wong Zhao Wu, Bryan; M. Faqih Akmal [@aqilakmal](https://github.com/aqilakmal); Oh Tien Cheng [@Tien-Cheng](https://github.com/Tien-Cheng)
Presentation Recording: https://youtu.be/47EwA529fqU

## Project Goal
The goal of the project is to create a Reinforcement Learning model that can land an agent in the Lunar Lander gym environment.

## Conclusion

| Model Name | Average Reward (↑) | Average Episode Length (↓) | Times Landed (%) (↑) |
|---|---|---|---|
|DQN|247.064|252.553 (best)|95.4%|
|DDQN|252.029 (best)|357.852|98.8% (best)|
|SARSA|245.834|391.061|84.5%|

<br>

In conclusion, it appears that the **Double DQN model performed the best**, attaining a higher average reward and success rate in landing, showing the benefit of improved Q value estimation. It appears to perform more cautiously than the DQN model, which may account for the improved performance. Ultimately, with the lowest failure rate of 1.2%, it shows itself as the best model for actually landing the Lunar Lander given the importance of a low failure rate in a high risk environment like landing a rocket autonomously. 

Hence, if we wanted to improve further on our results, an avenue of approach would be to improve the Double DQN further, adding improvements like a Prioritized Experience Replay Buffer, replacing epsilon-greedy exploration with noisy exploration, changing to a dueling architecture etc, which could result in even better performance.