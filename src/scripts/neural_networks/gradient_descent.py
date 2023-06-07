import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)
num_samples = 1000

x = torch.randn(num_samples, 2)

true_weights = torch.tensor([1.3, -1])
true_bias = torch.tensor([-3.5])

y = x @ true_weights.T + true_bias

fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].scatter(x[:, 0], y)
ax[1].scatter(x[:, 1], y)

ax[0].set_xlabel('X1')
ax[0].set_ylabel('Y')
ax[1].set_xlabel('X2')
ax[1].set_ylabel('Y')
plt.show()
