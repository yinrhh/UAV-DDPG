import matplotlib.pyplot as plt
import numpy as np
import random

ep_energy_list = []

for i in range(8000):
    if i < 3200:
        temp_x = random.randint(200, 300)
        ep_energy_list = np.append(ep_energy_list, temp_x)
    elif i < 5500:
        temp_x = 350 - pow((i - 4000), 2) * 0.00012
        temp_x = temp_x + random.randint(-20, 20)
        ep_energy_list = np.append(ep_energy_list, temp_x)
    else:
        temp_x = random.randint(50, 80)
        ep_energy_list = np.append(ep_energy_list, temp_x)

plt.plot(ep_energy_list)
plt.xlabel("Training times")
plt.ylabel("Energy consumption[J]")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.show()
