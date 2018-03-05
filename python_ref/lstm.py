# A long short-term memory model (LSTM) is a type of recurrent neural network
# that uses a gated memory cell to preserve hidden state information across
# long distances in sequence time.
# See publication at http://www.bioinf.at/publications/older/2604.pdf
# A good review can be found at
# https://pdfs.semanticscholar.org/06ba/e254319f8d39e80c7254c841787b45baf820.pdf

import sys
import multiprocessing
import threading
import math
import numpy
import scipy
# A helper class to map word or character based features into a feature index
class FeatureDict:
    def __init__(self):
        self.forward = dict()
        self.reverse = []
        self.lookup_word("START_OF_SEQUENCE", True)
        self.lookup_word("END_OF_SEQUENCE", True)

    # Given word (as string), return corresponding numerical index in dict
    # Set add_flag to True to automatically add to the dictionary if missing
    def lookup_word(self, word, add_flag):
        if word in self.forward:
            return self.forward[word]
        elif add_flag == True:
            self.forward[word] = len(self.forward)
            self.reverse.append(word)
            return self.forward[word]
    # Given numerical word index, return word as string
    def lookup_index(self, index):
        if index < len(self.reverse):
            return self.reverse[index]
        else:
            return ""

    # Returns the number of words in the dictionary, including the start and end of sequence words
    def get_vocab_size(self):
        return len(self.forward)

    # Given a string containing multiple characters, returns a list containing the seqeunce of numerical indexes accoding to the dict
    def make_char_sequence(self, line):
        sequence = []
        sequence.append(self.lookup_word("START_OF_SEQUENCE", True))
        char_list = list(line)
        for character in char_list:
            sequence.append(self.lookup_word(character, True))
            sequence.append(self.lookup_word("END_OF_SEQUENCE", True))
        return sequence

    # Given a string containing multiple words, returns a list containing the seqeunce of numerical indexes accoding to the dict
    def make_word_sequence(self, line):
        sequence = []
        sequence.append(self.lookup_word("START_OF_SEQUENCE", True))
        word_list = list(line)
        for word in word_list:
            sequence.append(self.lookup_word(word, True))
        sequence.append(self.lookup_word("END_OF_SEQUENCE", True))
        return sequence

    # Given a list containing numerical indexes, return string of corresponding characters
    def make_string_from_vector(self, sequence):
            string = ""
            for value in sequence:
                    if value > 1:
                            string += self.lookup_index(value)
            return string

# Apply non-linear activation function to vector
def activate(activation_type, vector):
        if activation_type == "logistic":
                for index in range(len(vector)):
                        if vector[index] > 10:
                                vector[index] = 1
                        elif vector[index] < -10:
                                vector[index] = 0
                        else:
                                vector[index] = 1/(1+math.exp(-vector[index]))
                return vector
                #return 1/(1+numpy.exp(-vector))
        elif activation_type == "softmax":
                max_val = numpy.amax(vector)
                prob = numpy.exp(vector-max_val)
                return prob/prob.sum()
        elif activation_type == "softplus":
                for index in range(len(vector)):
                        if vector[index] > 10:
                                pass
                        elif vector[index] < -10:
                                vector[index] = 0
                        else:
                                vector[index] = math.log(1+math.exp(vector[index]))
                return vector
        elif activation_type == "rectified linear":
                for index in range(len(vector)):
                        if vector[index] < 0:
                                vector[index] = 0
                return vector
        elif activation_type == "hyperbolic tangent":
                vector = numpy.tanh(vector)
                return vector
        else:
                return vector

# Evaluates the derivative of a non-linear activation function and multiplies it to the input vector (as in chain rule)
def chain_activation_derivative(activation_type, vector, other):
        if activation_type == "logistic":
                for index in range(len(vector)):
                        vector[index] = vector[index] * other[index] * (1-other[index])
                return vector
        elif activation_type == "rectified linear":
                for index in range(len(vector)):
                        if other[index] <= 0:
                                vector[index] = 0
                        elif vector[index] > 20: #gradient clipping
                                vector[index] = 20
                return vector
        elif activation_type == "hyperbolic tangent":
                for index in range(len(vector)):
                        vector[index] = vector[index] * (1-other[index]*other[index])
                return vector
        else:
                return vector

class LongShortTermMemory:
    def __init__(self, num_visible, num_hidden, num_output):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.num_gates = 3 # input, output, forget
        self.num_layers = len(num_hidden)
        weight_range = 1.0/math.sqrt(num_visible)
        numpy.random.seed(1)
        self.cell_weights = []
        for layer_index in range(self.num_layers):
            cell_weights = dict()
            num_inputs = self.num_visible
            if layer_index > 0:
                num_inputs = num_hidden[layer_index-1]
            num_outputs = num_hidden[layer_index]
            cell_weights["bottom"] = numpy.random.uniform(low=-1*weight_range, high=weight_range, size=(num_outputs * (1+self.num_gates), num_inputs))
            # visible to hidden weights matrix
            weight_range = 1.0/math.sqrt(num_outputs)
            # hidden to hidden weights matrix
            cell_weights["left"] = numpy.random.uniform(low=-1*weight_range, high=weight_range, size=(num_outputs * (1+self.num_gates), num_outputs))
            cell_weights["right"] = numpy.random.uniform(low=-1*weight_range, high=weight_range, size=(num_outputs * (1+self.num_gates), num_outputs))
            # visible to hidden bias vecto
            cell_weights["bias"] = numpy.zeros(num_outputs * (1+self.num_gates))
            #deltas for momentum calc
            cell_weights["bottom_velocity"] = numpy.zeros(cell_weights["bottom"].shape)
            cell_weights["left_velocity"] = numpy.zeros(cell_weights["left"].shape)
            cell_weights["bias_velocity"] = numpy.zeros(len(cell_weights["bias"]))
            self.cell_weights.append(cell_weights)
        # hidden to output weights matrix
        self.U = numpy.random.uniform(low=-1*weight_range, high=weight_range, size=(self.num_output, self.num_hidden[-1]))
        self.c = numpy.zeros(self.num_output)
        self.dU = numpy.zeros((self.num_output, self.num_hidden[-1]))
        self.dc = numpy.zeros(self.num_output)
        # hidden to output bias vector
        self.regularization_term = 0.000001
        self.final_momentum = 0.9
        self.initial_learning_rate = 0.1
        self.decay_rate = 5
        self.batch_size = 10
        self.dictionary = None
        # placeholder for peephole weights, currently unimplemented
        self.P = None

    # call before train() to set custom learning rate parameters
    def set_learning_rate(self, initial_rate, decay):
        self.initial_learning_rate = initial_rate
        self.decay_rate = decay

    # call before train() to set the dictionary -- needed to generate words
    def set_dictionary(self, dictionary):
        self.dictionary = dictionary

    def forward_cell(self, input_vector, left_cell, layer_index):
        outputs = dict()
        cell_weights = self.cell_weights[layer_index]
        # Equation numbers within parentheses in comments refer to Graves in [2]
        #g = Wx{t} + Vh{t-1} + b (4.2, 4.4, 4.6. 4.8)
        #pre_activation = visible.dot(self.W.T).ravel() + self.b
        pre_activation = numpy.copy(cell_weights["bias"])
        if layer_index == 0:
            pre_activation += cell_weights["bottom"][:,input_vector]
        else:
            pre_activation += cell_weights["bottom"].dot(input_vector)
        if left_cell != None and left_cell["output"] != []:
            pre_activation += cell_weights["left"].dot(left_cell["output"])
        if self.P != None and left_cell != None and left_cell["state"] != []:
            pre_activation += self.P.dot(left_cell["state"])
        #cell_input = activate("rectified linear", pre_activation[0:self.num_hidden]) # (4.7)

        num_hidden = self.num_hidden[layer_index]
        cell_input = activate("hyperbolic tangent", pre_activation[0:num_hidden]) # (4.7)
        input_gate = activate("logistic", pre_activation[num_hidden:2*num_hidden]) # (4.5)
        forget_gate = activate("logistic", pre_activation[2*num_hidden:3*num_hidden]) # (4.3)
        output_gate = activate("logistic", pre_activation[3*num_hidden:4*num_hidden]) # (4.9)

        # s{t} = s{t-1} ** f + c ** i (4.7)
        cell_state = cell_input * input_gate #hadamard product
        if left_cell != None and left_cell["state"] != []:
            cell_state += left_cell["state"] * forget_gate

        cell_state_copy = numpy.copy(cell_state)
        #h{t} = tanh(s{t}) ** o (4.10)
        #activated_cell_state = activate("rectified linear", cell_state_copy)
        activated_cell_state = activate("hyperbolic tangent", cell_state_copy)
        #activated_cell_state = activate("logistic", cell_state_copy)
        cell_output = activated_cell_state * output_gate
        outputs["input"] = cell_input
        outputs["state"] = cell_state
        outputs["activated_state"] = activated_cell_state
        outputs["input_gate"] = input_gate
        outputs["forget_gate"] = forget_gate
        outputs["output_gate"] = output_gate
        outputs["output"] = cell_output
        return outputs

    # forward pass through network for a single timestamp
    def forward_single(self, visible, prev_outputs):
        outputs = dict()
        outputs["cells"] = []
        left_cell = None
        for layer_index in range(self.num_layers):
            if prev_outputs != None and len(prev_outputs["cells"]) > 0:
                left_cell = prev_outputs["cells"][layer_index]
            bottom_vector = visible
            if layer_index > 0:
                bottom_vector = outputs["cells"][layer_index-1]["output"]
            current_cell_outputs = self.forward_cell(bottom_vector, left_cell, layer_index)
            outputs["cells"].append(current_cell_outputs)
            #y{t} = softmax(Uh{t} + c)
        classification = self.U.dot(current_cell_outputs["output"]) + self.c
        classification = activate("softmax", classification)
        outputs["classification"] = classification
        return outputs

    # forward pass through sequence of observations, from left to right to calculate outputs via inference
    def forward_pass(self, input_sequence):
        prev_output = None
        output_sequence = []
        for visible in input_sequence:
            output = self.forward_single(visible, prev_output)
            output_sequence.append(output)
            prev_output = output
        return output_sequence

    def backward_cell(self, error_vector, current_cell_outputs, left_cell_outputs, right_cell_outputs, right_cell_deltas, layer_index):
        deltas = dict()
        #d_h = U_T . d_y + V_T + d_g{t+1} (4.11)
        delta_output = error_vector
        if right_cell_deltas != None:
            delta_output += self.cell_weights[layer_index]["left"].T.dot(right_cell_deltas["combined"])
        deltas["output"] = delta_output

        #d_o = d_h * A(s_t) * sigm^(g_t) (4.12)
        delta_output_gate = chain_activation_derivative("logistic", delta_output * current_cell_outputs["activated_state"], current_cell_outputs["output_gate"])
        deltas["output_gate"] = delta_output_gate

        #d_s = d_h * o * A^(s_t) + f[t+1] * d_s[t+1] (4.13, sans peephole weights)
        #delta_state = chain_activation_derivative("rectified linear", delta_output * current_cell_outputs["output_gate"], current_cell_outputs["activated_state"])
        delta_state = chain_activation_derivative("hyperbolic tangent", delta_output * current_cell_outputs["output_gate"], current_cell_outputs["activated_state"])
        deltas["state"] = delta_state
        if right_cell_deltas != None:
            delta_state += right_cell_outputs["forget_gate"] * right_cell_deltas["state"]

        #d_c = d_s * i * A^(c_t)
        #delta_cell_input = chain_activation_derivative("rectified linear", delta_state * current_cell_outputs["input_gate"], current_cell_outputs["input"])
        delta_cell_input = chain_activation_derivative("hyperbolic tangent", delta_state * current_cell_outputs["input_gate"], current_cell_outputs["input"])
        deltas["input"] = delta_cell_input

        delta_state_forget = numpy.copy(delta_state)
        if left_cell_outputs != None:
            delta_state_forget = delta_state * left_cell_outputs["state"]
        delta_forget_gate = chain_activation_derivative("logistic", delta_state_forget, current_cell_outputs["forget_gate"])
        deltas["forget_gate"] = delta_forget_gate

        delta_input_gate = chain_activation_derivative("logistic", delta_state * current_cell_outputs["input"], current_cell_outputs["input_gate"])
        deltas["input_gate"] = delta_input_gate

        combined_input = delta_cell_input
        combined_input = numpy.concatenate((combined_input, delta_input_gate, delta_forget_gate, delta_output_gate))
        deltas["combined"] = combined_input
        return deltas

    # backward pass through sequence of observations, from right to left to calculate deltas via backpropagation
    def backward_pass(self, output_sequence, input_sequence):
        right_cell_deltas = None
        delta_sequence = []
        #for index, target in reversed(list(enumerate(target_sequence))):
        #reverse iterate
        for row_index in reversed(range(len(input_sequence)-1)):
            current_outputs = output_sequence[row_index]
            #d_y = x{t+1} - y{t}
            classification_error = -1 * current_outputs["classification"]
            target_value = input_sequence[row_index+1]
            classification_error[target_value] += 1
            deltas = dict()
            deltas["classification"] = classification_error
            deltas["cells"] = []
            current_delta = self.U.T.dot(classification_error)
            for layer_index in reversed(range(self.num_layers)):
                current_cell_outputs = current_outputs["cells"][layer_index]
                right_outputs = output_sequence[row_index+1]
                right_cell_outputs = right_outputs["cells"][layer_index]
                if len(delta_sequence) > 0:
                    right_cell_deltas = delta_sequence[0]["cells"][layer_index]
                else:
                    right_cell_deltas = None
                left_cell_outputs = None
                if row_index > 0:
                    left_outputs = output_sequence[row_index-1]
                    left_cell_outputs = left_outputs["cells"][layer_index]
                cell_deltas = self.backward_cell(current_delta, current_cell_outputs, left_cell_outputs, right_cell_outputs, right_cell_deltas, layer_index)
                current_delta = self.cell_weights[layer_index]["bottom"].T.dot(cell_deltas["combined"])
                deltas["cells"].insert(0, cell_deltas)
            delta_sequence.insert(0, deltas)
            right_cell_deltas = cell_deltas
        return delta_sequence

    # apply regularization term to avoid overfitting, implemented as L2 regularization (weight decay)
    def apply_regularization(self, learning_rate):
        for layer_index in range(self.num_layers):
            cell_weights = self.cell_weights[layer_index]
            cell_weights["bottom"] *= (1.0 - (self.regularization_term * learning_rate))
            cell_weights["left"] *= (1.0 - (self.regularization_term * learning_rate))
            cell_weights["bias"] *= (1.0 - (self.regularization_term * learning_rate))
            self.U *= (1.0 - (self.regularization_term * learning_rate))
            self.c *= (1.0 - (self.regularization_term * learning_rate))

    # update weights and bias parameters according to the gradient values calculated in the forward and backward passes
    def apply_gradient(self, step_size, inputs, outputs, deltas):
        # self.check_gradients(inputs, outputs, deltas)
        for seq_index in range(len(deltas)):
            current_cell_outputs = outputs[seq_index]["cells"][self.num_layers-1]
            self.U += step_size * numpy.outer(deltas[seq_index]["classification"], current_cell_outputs["output"] )
            self.c += step_size * deltas[seq_index]["classification"]
            # momentum updates
            if self.final_momentum > 0:
                self.dU += step_size * numpy.outer(deltas[seq_index]["classification"], current_cell_outputs["output"] )
                self.dc = step_size * deltas[seq_index]["classification"]
            for layer_index in range(self.num_layers):
                cell_weights = self.cell_weights[layer_index]
                current_cell_outputs = outputs[seq_index]["cells"][layer_index]
                current_cell_deltas = deltas[seq_index]["cells"][layer_index]
                combined_deltas = current_cell_deltas["combined"]
                left_cell_outputs = None
                if seq_index > 0:
                    left_cell_outputs = outputs[seq_index-1]["cells"][layer_index]
                bottom_vector = numpy.zeros(self.num_visible)
                if layer_index == 0:
                    bottom_vector[inputs[seq_index]] = 1
                else:
                    bottom_vector = outputs[seq_index]["cells"][layer_index-1]["output"]
                cell_weights["bottom"] += step_size * numpy.outer(combined_deltas, bottom_vector)
                if seq_index > 0:
                    cell_weights["left"] += step_size * numpy.outer(combined_deltas, left_cell_outputs["output"])
                cell_weights["bias"] += step_size * combined_deltas
                if self.final_momentum > 0:
                    cell_weights["bottom_velocity"] += step_size * numpy.outer(combined_deltas, bottom_vector)
                    if seq_index > 0:
                        cell_weights["left_velocity"] += step_size * numpy.outer(combined_deltas, left_cell_outputs["output"])
                    cell_weights["bias_velocity"] += step_size * combined_deltas

    def check_gradient(self, sequence, weights_matrix, derived_gradient_matrix, name):
        epsilon = 0.00001
        for row_index in range(weights_matrix.shape[0]):
            for col_index in range(weights_matrix.shape[1]):
                orig_weight = weights_matrix[row_index][col_index]
                weights_matrix[row_index][col_index] += epsilon
                objective_up = self.calc_objective(sequence)
                weights_matrix[row_index][col_index] -= 2*epsilon
                objective_down = self.calc_objective(sequence)
                weights_matrix[row_index][col_index] = orig_weigh
                numerical_gradient = (objective_down - objective_up)/(2*epsilon)
                derived_gradient = derived_gradient_matrix[row_index][col_index]
                if math.fabs(derived_gradient - numerical_gradient) > 0.1 and numerical_gradient != 0:
                    print(name + "[{}][{}]: num grad: {}, derived grad: {}, ratio: {}".format(row_index, col_index, numerical_gradient, derived_gradient, derived_gradient/numerical_gradient) )
        return

    # numerical check of gradient via brute force -- only for debugging purposes
    def check_gradients(self, sequence, outputs, deltas):
        grad_U = numpy.zeros(self.U.shape)
        layer_index = self.num_layers - 1
        for seq_index in range(len(deltas)):
            grad_U += numpy.outer(deltas[seq_index]["classification"], outputs[seq_index]["cells"][layer_index]["output"] )
            self.check_gradient(sequence, self.U, grad_U, "U")

        for layer_index in range(self.num_layers):
            cell_weights = self.cell_weights[layer_index]
            bottom_grad = numpy.zeros(cell_weights["bottom"].shape)
            left_grad = numpy.zeros(cell_weights["left"].shape)
            for seq_index in range(len(deltas)):
                combined_deltas = deltas[seq_index]["cells"][layer_index]["combined"]
                input_vector = numpy.zeros(self.num_visible)
                if layer_index == 0:
                    input_vector[sequence[seq_index]] = 1
                else:
                    input_vector = outputs[seq_index]["cells"][layer_index-1]["output"]
                bottom_grad += numpy.outer(combined_deltas, input_vector)
                if seq_index > 0:
                    left_grad += numpy.outer(combined_deltas, outputs[seq_index-1]["cells"][layer_index]["output"])
                self.check_gradient(sequence, cell_weights["bottom"], bottom_grad, "W"+str(layer_index))
                self.check_gradient(sequence, cell_weights["left"], left_grad, "V"+str(layer_index))

    # calculates and returns the objective function as the sum of cross-entropy losses, omitting the regularization term
    def calc_objective(self, sequence):
        objective = 0
        output_sequence = self.forward_pass(sequence)
        # for row_index in range(sequence.shape[0]-1):
        # target_coo_matrix = scipy.sparse.coo_matrix(sequence.getrow(row_index+1))
        # for row, col, value in zip(target_coo_matrix.row, target_coo_matrix.col, target_coo_matrix.data):
        for sequence_index in range(len(sequence)-1):
            target_value = sequence[sequence_index+1]
            target_prob = output_sequence[sequence_index]["classification"][target_value]
            if target_prob > 0:
                objective -= math.log(target_prob)
        return objective

    # updates the weight, bias, and velocity paramters in a Nesterov momentum type of algorithm. Velocities are calculated during apply gradient
    def apply_momentum(self, momentum):
        if momentum > 0:
            for layer_index in range(self.num_layers):
                cell_weights = self.cell_weights[layer_index]
                cell_weights["bottom_velocity"] *= momentum
                cell_weights["left_velocity"] *= momentum
                cell_weights["bias_velocity"] *= momentum
                cell_weights["bottom"] += cell_weights["bottom_velocity"]
                cell_weights["left"] += cell_weights["left_velocity"]
                cell_weights["bias"] += cell_weights["bias_velocity"]
            self.dU *= momentum
            self.dc *= momentum
            self.U += self.dU
            self.c += self.dc

    # randomly generate a sequence according to the model
    def generate(self):
        prev_outputs = None
        output_sequence = []
        visible = 0
        #start of sequence
        while visible != 1 and len(output_sequence) < 200:
            outputs = self.forward_single(visible, prev_outputs)
            next_char = numpy.random.choice(self.num_visible, 1, p=outputs["classification"])[0]
            #next_char = numpy.argmax(outputs["output"])
            output_sequence.append(next_char)
            visible = next_char
            prev_outputs = outputs
            # if len(output_sequence) < 4: #This may be a valid sequence, but too short to be interesting for evaluation purposes
            # return self.generate()
            # else:
        return output_sequence

    def run_iterations(self, sequences, num_iterations, step_size, momentum):
        num_sequences = len(sequences)
        for iteration in range(num_iterations):
            batch_array = numpy.random.permutation(num_sequences)[:self.batch_size]
            for sequence_index in batch_array:
                sequence = sequences[sequence_index]
                outputs = self.forward_pass(sequence)
                deltas = self.backward_pass(outputs, sequence)
                self.apply_momentum(momentum)
                self.apply_gradient(step_size, sequence, outputs, deltas)
            self.apply_regularization(step_size)

    # main training loop
    def train(self, sequences, max_epochs):
        #num_sequences = sequences.shape[0]
        num_sequences = len(sequences)
        print("num sequences: " + str(num_sequences) + ", vocab size: " + str(self.num_visible))
        num_iterations = int(num_sequences/self.batch_size)+1
        for epoch in range(max_epochs):
            sys.stdout.write(str(epoch) + "/" + str(max_epochs) + " : ")# + ", learning rate: " + str(learning_rate))
            learning_rate = self.initial_learning_rate / (1.0+(float(self.decay_rate*epoch))/float(max_epochs))
            momentum = self.final_momentum*math.tanh(epoch)
            sum_objective = 0
            for sequence in sequences:
                sum_objective += self.calc_objective(sequence)/len(sequence)
            sys.stdout.write("Obj={0:.4f} ({1:.2f}); ".format(sum_objective/num_sequences, math.exp(sum_objective/num_sequences)))
            generated = self.generate()
            print(self.dictionary.make_string_from_vector(generated))

            if True:
                self.run_iterations(sequences, num_iterations, learning_rate/self.batch_size, momentum)
            else:
                num_threads = multiprocessing.cpu_count()
                thread_list = []
                for thread_index in range(num_threads):
                    thread_obj = threading.Thread(target=self.run_iterations, args=(sequences, int(num_iterations/num_threads), learning_rate/self.batch_size, momentum))
                    thread_list.append(thread_obj)
                    thread_obj.start()
                for thread_obj in thread_list:
                    thread_obj.join()

    def save(self, base_filename):
        numpy.savez(base_filename, U=self.U, c=self.c, num_layers=self.num_layers)
        for layer_index in range(self.num_layers):
            cell_weights = self.cell_weights[layer_index]
            numpy.savez(base_filename + "." + str(layer_index), bottom=cell_weights["bottom"], left=cell_weights["left"], bias=cell_weights["bias"])

    def load(self, base_filename):
        try:
            data = numpy.load(base_filename + ".npz")
            self.U = data["U"]
            self.c = data["c"]
            self.num_layers = data["num_layers"]
            for layer_index in range(self.num_layers):
                data = numpy.load(base_filename + "." + str(layer_index) + ".npz")
                self.cell_weights[layer_index]["bottom"] = data["bottom"]
                self.cell_weights[layer_index]["left"] = data["left"]
                self.cell_weights[layer_index]["bias"] = data["bias"]
        except:
            pass

# Use own source code as input for training as a demonstration case
def test_on_self():
   ##tHiS cOmMeNt LinE should stand oUt aS ANOMALOUS!#
    num_hidden = [64, 48]
    num_epochs = 256
    sequences = []
    dictionary = FeatureDict()
    with open("lstm.py") as file_handle:
    #with open("hello.txt") as file_handle
        for line in file_handle:
            line = line.rstrip()
            #sequences.append(dictionary.make_char_sequence(line))
            if len(line) > 0:
                sequence = dictionary.make_word_sequence(line)
                sequences.append(sequence)
            #split by paragraph instead?
    vocab_size = dictionary.get_vocab_size()
    lstm = LongShortTermMemory(vocab_size, num_hidden, vocab_size)
    lstm.set_learning_rate(0.01, 10)
    lstm.set_dictionary(dictionary)
    lstm.load("self_test")
    lstm.train(sequences, num_epochs)
    lstm.save("self_test")
    log_likelihoods = []
    for sequence in sequences:
        log_likelihood = lstm.calc_objective(sequence)
        if len(sequence) > 2:
            log_likelihoods.append((log_likelihood-2)/(math.log(len(sequence)-2, 2)))
        else:
            log_likelihoods.append(log_likelihood-2)
    sorted_indices = numpy.argsort(log_likelihoods)
    for sorted_index in sorted_indices:
            print(str(log_likelihoods[sorted_index]) + " : " + dictionary.make_string_from_vector(sequences[sorted_index]))

# in case you want to run as a script rather than importing as a module, it can test itself
if __name__ == "__main__":
    test_on_self()
