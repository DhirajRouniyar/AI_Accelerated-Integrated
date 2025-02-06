# Output

![Alt text](https://github.com/DhirajRouniyar/AI_Accelerated-Integrated/blob/main/Low_Rank%20Approximation/Images/Output.png)
![Alt text](https://github.com/DhirajRouniyar/AI_Accelerated-Integrated/blob/main/Low_Rank%20Approximation/Images/Output_1.png)

# 1. What is a Low-Rank Neural Network?
A low-rank neural network is a compressed version of a regular neural network. Instead of using large weight matrices, it breaks them into smaller matrices that approximate the same function but with fewer parameters. This reduces the model size and speeds up training and inference.

Think of it like this:

A regular neural network stores everything in a large, complex table (matrix).
A low-rank neural network breaks that table into smaller pieces that approximate the same information, saving space and computation.

# 2. The Problem with Training Low-Rank Networks
Training a low-rank model requires setting a factorization rank for each layer, which controls how much compression is applied.

Too high a rank → The model is large and slow.
Too low a rank → The model loses accuracy.
Finding the right rank is difficult and requires trial and error.

# 3. What can be done?

Start with a full-rank model:

Instead of training with low-rank layers from the beginning, we train a regular model for a few epochs.
Monitor the "Stable Rank": For instance use CUTTLEFISH.

During training, CUTTLEFISH calculates the stable rank of each layer, which is an approximation of its true rank.
The stable rank gives an idea of how much compression each layer can handle without losing accuracy.
Switch to Low-Rank Training:

Once the stable rank stabilizes (i.e., it stops changing), CUTTLEFISH automatically replaces the full-rank layers with their low-rank approximations.
Each layer is compressed based on its stable rank, ensuring that no information is lost unnecessarily.
