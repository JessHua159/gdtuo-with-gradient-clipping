# gdtuo-with-gradient-clipping
This is modification of the gdtuo source code from https://github.com/kach/gradient-descent-the-ultimate-optimizer with an implementation of norm-based averaged gradient clipping as outlined in the following paper by Pascanu et al.: https://proceedings.mlr.press/v28/pascanu13.pdf.

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
Open ```sample_usage.ipynb``` in JupyterLab. In cell 3, uncomment the desired optimizer stack and leave the others commented. Optimizers with ```clip=True``` have gradient clipping enabled. Otherwise, gradient clipping is disabled. Execute the cells in the notebook as usual.
