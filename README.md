# Gradient Descent: The Ultimate Optimizer With Gradient Clipping
This repo contains:
- An extension of the gdtuo source code from https://github.com/kach/gradient-descent-the-ultimate-optimizer with an implementation of norm-based averaged gradient clipping as outlined in the following paper by Pascanu et al.: https://proceedings.mlr.press/v28/pascanu13.pdf.
- A [research paper](https://github.com/JessHua159/gdtuo-with-gradient-clipping/blob/main/research_paper.pdf) that describes the extension and the research conducted on the extension.
- Sample JupyterLab notebooks of the extension used for a [multi-layer perceptron (MNIST)](https://github.com/JessHua159/gdtuo-with-gradient-clipping/blob/main/sample_usage_mnist.ipynb) and a [convolutional neural network (CIFAR10)](https://github.com/JessHua159/gdtuo-with-gradient-clipping/blob/main/sample_usage_cifar10.ipynb).

# Abstract
Gradient descent is a popular and widely used optimization algorithm for training machine learning models. Within gradient descent, hyperparameters such as the step size, or learning rate, greatly affect the performance and convergence of the model and thus have a great impact. One way to determine these hyperparameters is by calculating a hypergradient to determine the optimal parameters from sub-optimal values. However, this method is prone to exploding gradients, and thus cannot handle initially high-valued hyperparmeters. In this study, we enhance the methodology to effectively handle high-valued initial hyperparameter values by employing the gradient clipping technique.

Our approach involves the incorporation of the normed-based averaged gradient clipping into the hyperoptimizer’s secondary optimization process. This modification, integrated into existing frameworks, contributes to mitigating the adverse effects of exploding gradients and improving overall system performance. The empirical evaluation, conducted on the MNIST dataset, demonstrates the efficacy of this enhancement, particularly in scenarios where traditional methods struggle with large hyperparameter values.

# Gradient Clipping Algorithm
$t = 1$: $\theta_t \leftarrow ||\mathbf{g}_t||_2$

$t > 1$: $\displaystyle \theta_t \leftarrow \frac{(\theta_{t - 1} * (t - 1)) + ||\mathbf{g}_t||_2}{t}$

For any $t$:

$\displaystyle \mathbf{g}_t' \leftarrow \mathbf{g}_t * \min\left(1, \frac{\theta_t}{||\mathbf{g}_t||_2}\right)$

$\mathbf{g}_t \leftarrow \mathbf{g}_t'$

$\displaystyle \theta_t' \leftarrow \frac{(\theta_{t - 1} * (t - 1)) + ||\mathbf{g}_t'||_2}{t}$

$\theta_t \leftarrow \theta_t'$

$t$ refers to the $t$-th step of the optimization algorithm across a minibatch.

# Usage
Open ```sample_usage_mnist.ipynb``` or ```sample_usage_cifar10.ipynb``` in JupyterLab. In cell 3, uncomment the desired optimizer stack and leave the others commented. Optimizers with ```clip=True``` have gradient clipping enabled. Otherwise, gradient clipping is disabled. Execute the cells in the notebook as usual.
