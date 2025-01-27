import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

KNOTS_VS_LOSS_PATH = Path("/ISTA---manifolds/model_analysis/knots_vs_loss.csv")
OUTPUT_PATH = Path("/ISTA---manifolds/model_analysis/knots_vs_loss.pdf")

# Step 2: Read the CSV file
df = pd.read_csv(KNOTS_VS_LOSS_PATH)

# Step 3: Select the columns to plot
knots = df['knots']
reg_weight = df['reg_weight']
train_loss = df['train_loss']
test_loss = df['test_loss']

# Step 4: Create the plot
plt.figure(figsize=(10, 6))
plt.plot(reg_weight, train_loss, '.', label='train loss')
plt.plot(reg_weight, test_loss, '.', label='test loss')

# Add titles and labels
plt.title('Train and test loss vs knots')
plt.xlabel('Reg weight')
plt.ylabel('L1 oss')
plt.xscale('log')
plt.grid(True)
plt.legend()

# Rotate x-axis labels
plt.xticks(rotation=45)

# Step 5: Save the plot as a PDF
plt.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight')
print(f"Saved plot to {OUTPUT_PATH}")
