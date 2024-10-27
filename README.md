# Flow-Matching for Generative Modeling Reproduction

This repository contains our implementation of a reproduction of the Flow-Matching approach for generative modeling. This work, inspired by the recent paper "[Flow Matching for Generative Modeling](https://arxiv.org/pdf/2210.02747)," aims to refine and enhance the generative model training process, improving speed and stability.

## Overview

In this project, we focus on training generative models without simulating full Ordinary Differential Equation (ODE) trajectories at each training step. By utilizing flow-matching techniques that maintain trajectory boundaries, we can streamline sampling from target distributions. Our approach also leverages Conditional Flow Matching (CFM) to create a more tractable loss function, enabling efficient learning of flow trajectories.

### Key Concepts

- **Trajectory Straightening**: By aligning flow trajectories to be straighter, we reduce the complexity of sampling, aiming for faster and higher-quality generation.
- **Conditional Flow Matching (CFM)**: CFM is used to address the intractability of naive flow-matching by conditioning on additional variables, thus simplifying the loss calculation.
- **Simulation-Free Training**: This method does not require full ODE trajectory simulation, resulting in faster and more stable model training.

## Methodology

Our approach is built around optimizing the flow-matching objective with a conditional expectation formulation, expressed as:

$$L_{CFM}(\theta) = E_{t, q(x_1), p_t(x_t|x_1)} \| v_\theta(x_t, t) - u(x_t, t | x_1) \|^2$$

Where $v_\theta(x_t, t)$ represents the learned flow vector field, and $u(x_t, t | x_1)$ captures the conditional flow direction at each time step.

## Results

1. **Enhanced Training Stability**: Our Flow Matching Optimal Transport (FM-OT) method provides more stable training than traditional approaches.
2. **Faster Sampling**: By straightening trajectories, we enable quicker ODE solving, often possible with a single Euler step.
3. **Sample Quality**: While FM-OT promotes efficient sampling, it remains inconclusive whether straighter trajectories yield improved sample quality in scenarios with infinite resources, as suggested in other studies ([reference](https://arxiv.org/abs/2404.12940)).

### Visual Results

We compared our reproduction to the paper's original results, including checkerboard visualizations to showcase model performance.

## Reproduce


### Setup
To get started, clone this repository and ensure you have the necessary dependencies installed:

```bash
git clone https://github.com/daniil-shlenskii/modern-gen-ai-flow-matching-reproduction.git
cd modern-gen-ai-flow-matching-reproduction
```

### Configuration

Create a YAML configuration file (e.g., config.yaml) to specify all training parameters. Below is an example of a configuration file:

```yaml
# Example config.yaml
img_size: 256                  # Size of input images
in_channels: 3                 # Number of input channels (e.g., 3 for RGB images)
dataloader: 'path_to_dataloader' # Path to your custom dataloader
mode: 'train'                  # Mode (e.g., 'train', 'test', etc.)
timesteps: 1000                # Number of timesteps for training
device: 'cuda'                 # Device to use ('cpu' or 'cuda')
num_epochs: 50                 # Number of training epochs
lr: 0.001                      # Learning rate
save_path: 'path_to_save_model' # Path to save the trained model
```

### Training

To initiate the training process, use the following command with your configuration file:

```bash
python scripts/train.py --config path_to_your_config.yaml
```
Make sure the config.yaml file is correctly set up and all paths are valid.

### Usage Example
Here's a quick example to demonstrate how to train a model using a sample configuration file:

1. Create configs/sample_config.yaml with the parameters you need.
2. Run the training script:
  ```bash
  python scripts/train.py --config configs/sample_config.yaml
  ```
3. The model will train according to the parameters provided in the configuration file and save the results to the specified save_path.




## Team Contributions

- **Alexander Sharashvin**: Data collection and method evaluation
- **Artem Alexeev**: Training pipeline and architecture development
- **Kseniia Petrushina**: Conditional Flow Matching class implementation and inference
- **Daniil Shlenskii**: Team management and presentation preparation

## Acknowledgments

This project was conducted as part of Skoltech course Modern Generative AI (October 2024).
