# 2D NGRC
The experiments use parallel Next Generation Reservoir Computing (NG RC) to predict the evolution of the 2D Kolmogorov Flow. The input variables are divided into different groups according to their spatial location. The input states of each group are predicted in parallel with a single Reservoir Computer.

## Setup
The Conda virtual environment was set up with the following installations:
-Python 3.10.9
-Numpy 1.23.5
-Matplotlib 3.6.2
-Scipy 1.9.3

## Parallel Architectures
The Multiprocessing tool is used to process tasks in parallel. In particular, the one step prediction of a specific spatial region is parallelized.

## Datasets
The training and test datasets have been generated based on the GitHub repo: https://github.com/google/jax-cfd. 
The dataset, combined in one matrix with the shape (16,2,3477,64,64) consists of: 
- 16 trajectories
- 2 properties of the fluid. The velocity of the fluid into two directions
- 3477 time steps
- A resolution of 64 spatial points into one direction
- A resolution of 64 spatial points into the other direction
  
The first and 15th trajectories are used to train the model. The 16th trajectory is used as a test trajectory. 
 
## Quick Start 
1. Generate the training and test data using the referenced Github repository
2. Save “Kolmogorov_All_Trajectories.h5” to the desired location
3. Update the data path in the main file to access the data
4. Set up the virtual environment
5. Run the main file
6. The main file will predict the evolution of the Kolmogorov Flow, evaluate the predictive perfromance and print a graphical interpretation of the prediction:
   ![Noise400, Ridge para0 0090088, past_states2, Groupsize1, Interaction2, Pred_lengh10, scalingFalse, usedTrajectories1, onGroupnumbers15, ausgewaehlte Groupnumbers](https://github.com/JohannesSautier/2D_NGRC/assets/114988239/987f4aa8-67ab-4e00-86b0-b1890fcf1a13)

