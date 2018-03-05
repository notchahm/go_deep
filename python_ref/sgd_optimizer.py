"""
sgd_optimizer.py

Stochastic gradient descent optimizer with Nesterov momentum and Hogwild-style
updates

Author: Chahm An
notchahm@gmail.com
"""

class StochasticGradientDescentOptimizer():
    def __init__(self):
        self.learning_rate = 0.1
        self.learning_rate_decay = 50.0
        self.regularization_rate = 0.1
        self.batch_size = 10

    def train_epoch(self, epoch, training_set, model):
        num_iterations = training_set.get_num_images()
        step_size = self.learning_rate / (1.0 + epoch / self.learning_rate_decay)
        for iteration in range(num_iterations):
            batch = training_set.randomly_sample_batch(self.batch_size)
            for image, target in batch:
                network_outputs = model.calculate_output(image)
                loss, error_delta = model.calculate_loss(network_outputs[-1], target)
                network_deltas = model.calculate_delta(error_delta, network_outputs)
                model.apply_gradient(step_size, network_deltas, network_outputs, image);
            weight_decay_rate = self.regularization_rate * step_size * self.batch_size;
            model.apply_regularization(weight_decay_rate);

