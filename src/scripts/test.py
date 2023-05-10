import random
import matplotlib.pyplot as plt
from matplotlib import style

# x, y = np.random.random(size=(2, 10))
# print(x)
# print(y)

style.use("dark_background")
plt.axis('off')

for i in range(500):
    p1 = random.randint(-1000, 1000)
    plt.plot(
        [p1, random.randint(-1000, 1000)],
        [p1, random.randint(-1000, 1000)],
        'ro-',
        linewidth=0.3,
        markersize=0.5,
        color="orange",
    )

plt.show()
