import matplotlib.pyplot as plt
import torch
from grimMethods import neuron
import numpy as np

class Neuron:
    def __init__(self, weights: torch.Tensor, bias: torch.Tensor, activation: str):
        self.weights = weights
        self.bias = bias
        self.activation = activation

        # Add safety checks
        if not isinstance(self.weights, torch.Tensor):
            raise TypeError("Weights must be a torch.Tensor")
        if not isinstance(self.bias, torch.Tensor):
            raise TypeError("Bias must be a torch.Tensor")
        if self.weights.dim() != 1:
            raise ValueError("Weights must be a 1D tensor")
        if self.bias.dim() != 0:
            raise ValueError("Bias must be a scalar tensor")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.weights + self.bias
        if self.activation == 'relu':
            return self.relu(z)
        elif self.activation == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation == 'softmax':
            return self.softmax(z)
        elif self.activation == 'tanh':
            return self.tanh(z)
        return z

    # Activation Functions
    def relu(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)  # or F.relu(x)

    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)  # or F.sigmoid(x)

    def softmax(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=1)  # or F.softmax(x, dim=1)

    def tanh(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)  # or F.tanh(x)

    # Class Methods
    def __repr__(self):
        return f"Neuron(activation={self.activation})"

    def __str__(self):
        return f"Neuron with {self.weights.shape[0]} inputs, activation={self.activation}"

    def _weights(self):
        print("Weights:", self.weights)

    def _bias(self):
        print("Bias:", self.bias)

    def _activation(self):
        print("Activation:", self.activation)

    def draw_neuron(self):
        num_inputs = len(self.weights)
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Input positions
        input_x = np.full(num_inputs, -2)
        y_span = max(2, num_inputs * 0.50)
        input_y = np.linspace(-y_span/2, y_span/2, num_inputs)
        
        # Dynamically scale vertical spacing based on number of inputs
        neuron_pos = (0, 0)
        act_pos = (2, 0)
        output_pos = (4, 0)

        # Draw arrows from inputs to neuron
        for i, (x, y) in enumerate(zip(input_x, input_y)):
            ax.annotate('', xy=neuron_pos, xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', lw=2, color='gray'), zorder=1)
            
        # Arrow from neuron to activation
        ax.annotate('', xy=act_pos, xytext=neuron_pos,
                    arrowprops=dict(arrowstyle='->', lw=2, color='gray'), zorder=1)
        
        # Put z label on top of the arrow
        mid_x_neuron_act = (neuron_pos[0] + act_pos[0]) / 2
        mid_y_neuron_act = (neuron_pos[1] + act_pos[1]) / 2
        ax.text(mid_x_neuron_act, mid_y_neuron_act + 0.18, "$z$", fontsize=12, ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='none', alpha=0.8), zorder=3)
        
        
        # Arrow from activation to output
        ax.annotate('', xy=output_pos, xytext=act_pos,
                    arrowprops=dict(arrowstyle='->', lw=2, color='gray'), zorder=1)

        # Put y label on top of the arrow
        mid_x_act_out = (act_pos[0] + output_pos[0]) / 2
        mid_y_act_out = (act_pos[1] + output_pos[1]) / 2
        ax.text(mid_x_act_out, mid_y_act_out + 0.18, "$y$", fontsize=12, ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='none', alpha=0.8), zorder=3)

        # Draw input circles using Circle patches instead of plot
        for i, (x, y) in enumerate(zip(input_x, input_y)):
            input_circle = plt.Circle((x, y), 0.20, color='lightgreen', zorder=2)
            ax.add_patch(input_circle)
            ax.text(x-0.5, y, f"$x_{{{i+1}}}$", fontsize=12, ha='right', va='center', zorder=3)
            ax.text(x, y, f"$w_{{{i+1}}}$", fontsize=12, ha='center', va='center', zorder=3)

        # Draw bias
        bias_circle = plt.Circle((neuron_pos[0], neuron_pos[1] + 1), 0.20, color='palegoldenrod', zorder=2)
        ax.add_patch(bias_circle)
        ax.annotate('', xy=neuron_pos, xytext=(neuron_pos[0], neuron_pos[1] + 1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='gray'), zorder=1)
        ax.text(neuron_pos[0], neuron_pos[1] + 1, "$b$", fontsize=12, ha='center', va='center', zorder=3)

        # Draw neuron circle (pre-activation z)
        circle = plt.Circle(neuron_pos, 0.3, color='skyblue', ec='skyblue', zorder=2)
        ax.add_patch(circle)

        # Draw activation circle (red)
        act_circle = plt.Circle(act_pos, 0.3, color='lightcoral', ec='lightcoral', lw=2, zorder=2)
        ax.add_patch(act_circle)
        ax.text(act_pos[0], act_pos[1] - 1,  f"${self.activation}$", fontsize=12, ha='center', va='center', zorder=3)

        # Add title
        title = f"$z = \\sum_{{i=1}}^{{{num_inputs}}} w_i x_i + b$ and $y = {self.activation}(z)$"
        ax.set_title(title, fontsize=14, pad=0)

        # Set plot limits
        ax.set_xlim(-3, 5)
        ax.set_ylim(-y_span, y_span)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.show()

    def _info(self):
        self._weights()
        self._bias()
        self._activation()

    if __name__ == "__main__":
        test_neuron = neuron.Neuron(weights=torch.randn(2), bias=torch.tensor(1.9), activation='tanh')
        print(test_neuron)
        test_neuron._info()  # Prints weights, bias, and activation
        test_neuron.draw_neuron()  # Draws the neuron graphically
        x = torch.randn(2)
        print("Input:", x)

        print("Forward pass output:", test_neuron.forward(x))

