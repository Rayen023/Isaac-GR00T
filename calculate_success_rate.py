import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('test_results_20251020_191220/results.csv')
print(f"Success Rate: {df['success'].mean() * 100:.2f}%")
positions = pd.read_csv('object_positions.csv')
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(positions['block_x'][df['success'] == 1], positions['block_y'][df['success'] == 1], c='green', label='Success', alpha=0.6)
plt.scatter(positions['block_x'][df['success'] == 0], positions['block_y'][df['success'] == 0], c='red', label='Failure', alpha=0.6)
plt.xlabel('Block X'); plt.ylabel('Block Y'); plt.legend(); plt.title('Block Positions: Success vs Failure'); plt.grid(True)
plt.subplot(1, 2, 2)
plt.scatter(positions['cup_x'][df['success'] == 1], positions['cup_y'][df['success'] == 1], c='green', label='Success', alpha=0.6)
plt.scatter(positions['cup_x'][df['success'] == 0], positions['cup_y'][df['success'] == 0], c='red', label='Failure', alpha=0.6)
plt.xlabel('Cup X'); plt.ylabel('Cup Y'); plt.legend(); plt.title('Cup Positions: Success vs Failure'); plt.grid(True)
plt.tight_layout(); plt.show()

# save the plots
plt.savefig('success_failure_positions.png')
