import matplotlib.pyplot as plt

filepath = r"src\backend\results\MLP\ml100k\loss_results\mf.txt"

training_losses = []
validation_losses = []

with open(filepath, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("trainingLoss") or line.startswith("Training time"):
            continue
        train, val = line.split(",")
        training_losses.append(float(train.strip()))
        validation_losses.append(float(val.strip()))

plt.figure(figsize=(12, 8))
plt.plot(training_losses, label="Training Loss", linewidth=3)
plt.plot(validation_losses, label="Validation Loss", linewidth=3)

x_pos = len(training_losses) - 6
y_train = training_losses[x_pos]
y_val = validation_losses[x_pos]
plt.vlines(
    x=x_pos,
    ymin=y_train,
    ymax=y_val,
    color='red',
    linestyle='--',
    linewidth=2,
    label='Best Validation Loss'
)

plt.xlabel("Epoch", fontsize=28, fontweight='bold')
plt.ylabel("Loss", fontsize=28, fontweight='bold')
plt.title("Training vs Validation Loss", fontsize=24, fontweight='bold')
plt.legend(prop={'size': 24, 'weight': 'bold'})
plt.xticks(fontsize=22, fontweight='bold')
plt.yticks(fontsize=22, fontweight='bold')
plt.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.7, color='gray')

plt.show()