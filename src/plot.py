import numpy as np
import matplotlib.pyplot as plt

bad_losses_MUTAG = np.load("../bad_losses_MUTAG.npy", allow_pickle=True)
clean_losses_MUTAG = np.load("../clean_losses_MUTAG.npy", allow_pickle=True)
clean_losses_AIDS = np.load("../clean_losses_AIDS.npy", allow_pickle=True)
bad_losses_AIDS = np.load("../bad_losses_AIDS.npy", allow_pickle=True)

clean_losses_ENZYMES = np.load("../clean_losses_ENZYMES.npy", allow_pickle=True)
bad_losses_ENZYMES = np.load("../bad_losses_ENZYMES.npy", allow_pickle=True)

fig, ax = plt.subplots(1, 3, figsize=(12, 2.5))

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 14
start = 2
end = 30
ax[0].plot(bad_losses_MUTAG[start:end, 0], 'r--')
ax[0].plot(clean_losses_MUTAG[start:end, 0], 'b--')
ax[0].fill_between(range(0, end-start), clean_losses_MUTAG[start:end, 1], clean_losses_MUTAG[start:end, 2], color="blue", alpha=0.25, edgecolor="blue")
ax[0].fill_between(range(0, end-start), bad_losses_MUTAG[start:end, 1], bad_losses_MUTAG[start:end, 2], color="red", alpha=0.25)
print(bad_losses_MUTAG)
print(clean_losses_MUTAG)
ax[0].grid()

ax[0].set_xticklabels([0, 0 ,15, 30])
ax[0].set_yticklabels([0, 0, 0.5, 1, 1.5])
ax[0].set_ylabel("Training Loss", fontsize=16, fontname = 'Times New Roman')
ax[0].legend(["Backdoor Samples", "Clean Samples"], fontsize=14)
ax[0].set_title("MUTAG", fontsize=16, fontname = 'Times New Roman')

# ax[0].set_xticks(color='w')
start = 0
end = 30
ax[1].plot(clean_losses_AIDS[start:end, 0], 'b--')
ax[1].plot(bad_losses_AIDS[start:end, 0], 'r--')
ax[1].fill_between(range(0, end-start), clean_losses_AIDS[start:end, 1], clean_losses_AIDS[start:end, 2], color="blue", alpha=0.25)
ax[1].fill_between(range(0, end-start), bad_losses_AIDS[start:end, 1], bad_losses_AIDS[start:end, 2], color="red", alpha=0.25)

ax[1].grid()
ax[1].set_xticklabels([0, 0 ,15, 30])
ax[1].set_yticklabels([0, 0, 0.5, 1, 1.5])
ax[1].set_title("AIDS", fontsize=16, fontname = 'Times New Roman')

start = 0
end = 100
print(bad_losses_ENZYMES[start:end, :])
ax[2].plot(clean_losses_ENZYMES[start:end, 0], 'b--')
ax[2].plot(bad_losses_ENZYMES[start:end, 0], 'r--')
ax[2].fill_between(range(0, end-start), clean_losses_ENZYMES[start:end, 1], clean_losses_ENZYMES[start:end, 2], color="blue", alpha=0.25)
ax[2].fill_between(range(0, end-start), bad_losses_ENZYMES[start:end, 1], bad_losses_ENZYMES[start:end, 2], color="red", alpha=0.25)

ax[2].grid()
ax[2].set_xticklabels([0, 0 ,50 ,100])
ax[2].set_ylim([0,4])
# ax[2].set_yticklabels([0, 0, 0.5, 1, 1.5])
ax[2].set_title("ENZYMES", fontsize=16, fontname = 'Times New Roman')

for i in range(3):
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)

# plt.plot(clean_losses_ENZYMES[1:15], 'r-x')
# plt.plot(bad_losses_ENZYMES[1:15], 'b-^')
plt.show()
fig.savefig("pre_experiment.pdf", bbox_inches='tight')