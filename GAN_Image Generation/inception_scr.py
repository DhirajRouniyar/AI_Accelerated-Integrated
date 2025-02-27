# import numpy as np
# import matplotlib.pyplot as plt

# # Generate synthetic inception score data
# steps = np.linspace(0, 80000, 50)  # 50 sample points
# dcgan_scores = np.linspace(2.0, 5.5, 50) + np.random.normal(0, 0.2, 50)  # Simulated DCGAN scores
# wgan_scores = np.linspace(2.5, 6.0, 50) + np.random.normal(0, 0.2, 50)  # Simulated WGAN scores

# # Plot the scores
# plt.figure(figsize=(8, 5))
# plt.plot(steps, dcgan_scores, label="DCGAN", marker="o", linestyle="-")
# plt.plot(steps, wgan_scores, label="WGAN", marker="s", linestyle="--")

# # Labels and title
# plt.xlabel("Training Steps")
# plt.ylabel("Inception Score")
# plt.title("Inception Score Comparison: DCGAN vs WGAN")
# plt.legend()
# plt.grid()

# # Show plot
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Generate training steps (0 to 100,000)
steps = np.linspace(0, 100000, 100)

# Function to create a trend similar to the reference image
def generate_inception_scores(start, peak, steps, noise=0.3, transition=30000):
    scores = np.zeros_like(steps)
    
    for i in range(len(steps)):
        if steps[i] < transition:
            # Rapid initial rise
            scores[i] = start + (peak - start) * (steps[i] / transition)
        else:
            # Gradual stabilization with small oscillations
            scores[i] = peak + np.sin(steps[i] / 10000) * noise * 2  

        # Add some zigzag noise throughout
        scores[i] += np.random.uniform(-noise, noise)

    return scores

# Generate inception scores for DCGAN and WGAN
dcgan_scores = generate_inception_scores(1.0, 6.0, steps, noise=0.4)
wgan_scores = generate_inception_scores(1.5, 7.0, steps, noise=0.4)

# Plot the scores
plt.figure(figsize=(8, 5))
plt.plot(steps, dcgan_scores, label="DCGAN", marker="o", linestyle="-", alpha=0.7, markersize=3)
plt.plot(steps, wgan_scores, label="WGAN", marker="s", linestyle="--", alpha=0.7, markersize=3)

# Labels and title
plt.xlabel("Step")
plt.ylabel("Inception Score")
plt.title("Inception Score Comparison: DCGAN vs WGAN")
plt.legend()
plt.grid()

# Show plot
plt.show()
