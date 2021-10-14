import random
import matplotlib.pyplot as plt
import numpy as np
import math

class Sample():
    def __init__(self, x1: int, x2: int, y: int):
        self.x1 = x1
        self.x2 = x2
        self.y = y

class PerceptronModel():
        # w0 - bias
    def __init__(self, w0: float, w1: float, w2: float, alfa: float, bias_enabled: bool, threshold: float):
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.alfa = alfa
        self.bias_enabled = bias_enabled
        self.threshold = threshold

class AdalineModel():
        # w0 - bias
    def __init__(self, w0: float, w1: float, w2: float, ni: float, tolerated_error: float, bias_enabled: bool, threshold: float):
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.ni = ni
        self.tolerated_error = tolerated_error
        self.bias_enabled = bias_enabled
        self.threshold = threshold

training_samples_unipolar = [Sample(0,0,0), Sample(0,1,0), Sample(1,0,0), Sample(1,1,1)]
training_samples_bipolar = [Sample(-1,-1,-1), Sample(-1,1,-1), Sample(1,-1,-1), Sample(1,1,1)]

def get_samples(unipolar: bool):
    if unipolar:
        return training_samples_unipolar
    return training_samples_bipolar

def init_perceptron_model(model):
    model.w0 = random.uniform(-0.3, 0.3)
    model.w1 = random.uniform(-0.3, 0.3)
    model.w2 = random.uniform(-0.3, 0.3)
    model.alfa = 0.3

def init_adaline_model(model):
    model.w0 = random.uniform(-0.3, 0.3)
    model.w1 = random.uniform(-0.3, 0.3)
    model.w2 = random.uniform(-0.3, 0.3)
    model.ni = 0.001
    model.tolerated_error = 0.3

                            #returns number of iterations it took to train a model
def train_model(model, training_samples, isUnipolar: bool, verboseMode=False) -> int:
    model_trained = False
    number_of_iterations = 0
    while not model_trained:
        error = process_collection(training_samples, model, isUnipolar, verboseMode=verboseMode)
        model_trained = not error
        number_of_iterations += 1
        if verboseMode:
            print(str(model.w0) + " " + str(model.w1) + " " + str(model.w2) + " iterations: " + str(number_of_iterations))
    return number_of_iterations


                            #returns number of iterations it took to train a model
def train_model_adaline(model, training_samples, iterations_limit, verboseMode=False) -> int:
    model_trained = False
    number_of_iterations = 0
    while not model_trained and number_of_iterations < iterations_limit:
        error = process_collection_adaline(training_samples, model, verboseMode=verboseMode)
        model_trained = error < model.tolerated_error
        number_of_iterations += 1
        if verboseMode:
            print(str(model.w0) + " " + str(model.w1) + " " + str(model.w2) + " square error = " + str(error) + " iterations: " + str(number_of_iterations))
    return number_of_iterations if number_of_iterations < iterations_limit else math.inf


def process_collection(collection, model, isUnipolar: bool, verboseMode=False) -> bool:
    error_occured = False
    for sample in collection:
        error_in_sample = process_sample(sample, model, isUnipolar, verboseMode=verboseMode)
        if error_in_sample:
            error_occured = True
    return error_occured

def process_collection_adaline(collection, model, verboseMode=False) -> float:
    total_error = 0.0
    for sample in collection:
        error_in_sample = process_sample_adaline(sample, model, verboseMode=verboseMode)
        total_error += error_in_sample
    return total_error / len(collection)

    #returns whether error occured or not
def process_sample(sample: Sample, model, isUnipolar: bool, verboseMode=False) -> bool:
    eval_y = evaluate_output(total_excitation(sample, model), model.threshold, isUnipolar)
    error = sample.y - eval_y   #expected - evaluated
    #update weights
    model.w1 = model.w1 + model.alfa * error * sample.x1
    model.w2 = model.w2 + model.alfa * error * sample.x2
    if model.bias_enabled:
        model.w0 = model.w0 + model.alfa * error * 1
    if verboseMode:
        print("\t" + str(model.w0) + " " + str(model.w1) + " " + str(model.w2) + " threshold= " + str(model.threshold))
    return False if error == 0 else True

    #returns square_error for processed sample
def process_sample_adaline(sample: Sample, model, verboseMode=False) -> float:
    total_excit = total_excitation(sample, model)
    error = error_for_sample(sample.y, total_excit, verboseMode=verboseMode)

    model.w1 = model.w1 + 2 * model.ni * error * sample.x1
    model.w2 = model.w2 + 2 * model.ni * error * sample.x2
    if model.bias_enabled:
        model.w0 = model.w0 + 2 * model.ni * error * 1

    return error * error


def total_excitation(sample: Sample, model) -> float:
    return model.w0 + model.w1 * sample.x1 + model.w2 * sample.x2

def evaluate_output(total_excitation_value: float, threshold: float, isUnipolar: bool) -> int:
    if isUnipolar:
        return 1 if total_excitation_value > threshold else 0
    else:
        return 1 if total_excitation_value > threshold else -1

def error_for_sample(expected, total_excitation, verboseMode=False):
    if verboseMode:
        print("\tError for sample = " + str(expected - total_excitation))
    return expected - total_excitation

def square_error_for_sample(expected, total_excitation):
    error = error_for_sample(expected, total_excitation)
    return error * error

def predict(x, y, w0, w1, w2, threshold, isUnipolar) -> int:
    if isUnipolar:
        return 1 if w0 + w1 * x + w2 * y > threshold else 0
    else:
        return 1 if w0 + w1 * x + w2 * y > threshold else -1

                                                            # return float: <0.0, 1.0>
def get_model_accuracy(model, test_set, prediction_function) -> float:
    wrong_predictions = 0
    for sample in test_set:
        if sample.y != prediction_function(model, sample):
            wrong_predictions += 1
    return (len(test_set) - wrong_predictions) / len(test_set)

def test_1_1():
    results  = []
    for t in (0.2, 0.4, 0.5, 0.6, 0.8, 1):
        iterations = []
        for i in range(10):
            w1 = random.uniform(-0.3,0.3)
            w2 = random.uniform(-0.3,0.3)
            current_model = PerceptronModel(w0=0, w1=w1, w2=w2, alfa=0.02, bias_enabled=False, threshold=t)
            unipolar = True
            training_samples = get_samples(unipolar)
            iterations.append(train_model(current_model, training_samples, unipolar))
        results.append((t, sum(iterations)/len(iterations)))
    print("\n\nRESULTS\n\n")
    for x in results:
        print("Value of threshold = " + str(x[0]) + " Average number of iterations = " + str(x[1]))

def test_1_2():
    results = []
    for range_of_weights in (1, 0.9, 0.8, 0.7, 0.6, 0.4, 0.2, 0.1, 0.05):
        iterations = []
        for i in range(10):
            w0 = random.uniform(-range_of_weights,range_of_weights)
            w1 = random.uniform(-range_of_weights,range_of_weights)
            w2 = random.uniform(-range_of_weights,range_of_weights)
            current_model = PerceptronModel(w0,w1,w2, alfa=0.2, bias_enabled=True, threshold=0)
            unipolar = True
            training_samples = get_samples(unipolar)
            iterations.append(train_model(current_model, training_samples, unipolar))
        results.append((range_of_weights, sum(iterations)/len(iterations)))
    print("\n\nRESULTS\n\n")
    for x in results:
        print("Range of weights random value = (-" + str(x[0]) + "," + str(x[0]) + ") Average number of iterations = " + str(x[1]))

def test_1_3():
    results = []
    for alfa in (1, 0.9, 0.8, 0.7, 0.6, 0.4, 0.2, 0.1, 0.05):
        iterations = []
        for i in range(10):
            w0 = random.uniform(-0.3,0.3)
            w1 = random.uniform(-0.3,0.3)
            w2 = random.uniform(-0.3,0.3)
            current_model = PerceptronModel(w0,w1,w2, alfa, bias_enabled=True, threshold=0)
            unipolar = True
            training_samples = get_samples(unipolar)
            iterations.append(train_model(current_model, training_samples, unipolar))
        results.append((alfa, sum(iterations)/len(iterations)))
    print("\n\nRESULTS\n\n")
    for x in results:
        print("Alfa value = " + str(x[0]) + " Average number of iterations = " + str(x[1]))

def test_1_4():
    results_unipolar = []
    results_bipolar = []
    for alfa in (1, 0.9, 0.8, 0.7, 0.6, 0.4, 0.2, 0.1, 0.05):
        iterations_uni = []
        iterations_bi = []
        for i in range(10):
            w0 = random.uniform(-0.3,0.3)
            w1 = random.uniform(-0.3,0.3)
            w2 = random.uniform(-0.3,0.3)

            #unipolar
            current_model = PerceptronModel(w0,w1,w2, alfa, True, threshold=0)
            unipolar = True
            training_samples = get_samples(unipolar)
            iterations_uni.append(train_model(current_model, training_samples, unipolar))

            #bipolar
            current_model = PerceptronModel(w0, w1, w2, alfa, True, threshold=0)
            unipolar = False
            training_samples = get_samples(unipolar)
            iterations_bi.append(train_model(current_model, training_samples, unipolar))
        results_unipolar.append((alfa, sum(iterations_uni)/len(iterations_uni)))
        results_bipolar.append((alfa, sum(iterations_bi)/len(iterations_bi)))

    print("\n\nRESULTS\n\n")

    print("RESULTS FOR UNIPOLAR")
    for x in results_unipolar:
        print("Alfa value = " + str(x[0]) + " Average number of iterations = " + str(x[1]))
    iteration_numbers = [x[1] for x in results_unipolar]
    print("Average = " + str(sum(iteration_numbers)/len(iteration_numbers)))

    print("RESULTS FOR BIPOLAR")
    for x in results_bipolar:
        print("Alfa value = " + str(x[0]) + " Average number of iterations = " + str(x[1]))
    iteration_numbers = [x[1] for x in results_bipolar]
    print("Average = " + str(sum(iteration_numbers)/len(iteration_numbers)))

def test_2_1():
    results = []
    for range_of_weights in (1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05):
        iterations = []
        for i in range(10):
            w0 = random.uniform(-range_of_weights,range_of_weights)
            w1 = random.uniform(-range_of_weights,range_of_weights)
            w2 = random.uniform(-range_of_weights,range_of_weights)
            current_model = AdalineModel(w0, w1, w2, ni=0.0001, tolerated_error=0.3, bias_enabled=True, threshold=0)
            unipolar = False
            training_samples = get_samples(unipolar)
            iterations.append(train_model_adaline(current_model, training_samples, iterations_limit=10000))
        results.append((range_of_weights, sum(iterations)/len(iterations)))
    print("\n\nRESULTS\n\n")
    for x in results:
        print("Range of weights random value = (-" + str(x[0]) + "," + str(x[0]) + ") Average number of iterations = " + str(x[1]))

def test_2_2():
    results = []
    for ni in (0.0001, 0.001, 0.01, 0.1, 0.2):
        iterations = []
        for i in range(10):
            w0 = random.uniform(-0.3,0.3)
            w1 = random.uniform(-0.3,0.3)
            w2 = random.uniform(-0.3,0.3)
            current_model = AdalineModel(w0,w1,w2, ni=ni, tolerated_error=0.3, bias_enabled=True, threshold=0)
            unipolar = False
            training_samples = get_samples(unipolar)
            iterations.append(train_model_adaline(current_model, training_samples, iterations_limit=10000))
        results.append((ni, sum(iterations)/len(iterations)))
    print("\n\nRESULTS\n\n")
    for x in results:
        print("Ni = " + str(x[0]) + " Average number of iterations = " + str(x[1]))

def test_2_3():
    results = []
    for tolerated_error in (0.1, 0.3, 0.5, 0.7, 1.0, 1.5):
        iterations = []
        accuracies = []
        for i in range(10):
            w0 = random.uniform(-0.3,0.3)
            w1 = random.uniform(-0.3,0.3)
            w2 = random.uniform(-0.3,0.3)
            current_model = AdalineModel(w0,w1,w2, ni=0.001, tolerated_error=tolerated_error, bias_enabled=True, threshold=0)
            unipolar = False
            training_samples = get_samples(unipolar)
            iterations.append(train_model_adaline(current_model, training_samples, iterations_limit=10000))
            accuracies.append(
                get_model_accuracy(current_model, training_samples, lambda model, sample: predict(  #checking accuracy on training set - in this situation used only to check a behaviour of algorithm
                    sample.x1, sample.x2, model.w0, model.w1, model.w2, model.threshold, unipolar)))
        results.append((tolerated_error, sum(iterations)/len(iterations), sum(accuracies)/len(accuracies)))
    print("\n\nRESULTS\n\n")
    for x in results:
        print("Tolerated error = " + str(x[0]) + " Average number of iterations = " + str(x[1]) + " Average accuracy of model = " + str(x[2]))


def execute_tests():
    print("Tests in execution!")
    print("\nTests for perceptron:\n")
    test_1_1()
    test_1_2()
    test_1_3()
    test_1_4()
    print("\nTests for Adaline:\n")
    test_2_1()
    test_2_2()
    test_2_3()

def __main__():
    execute_tests()

    print("Testing manually model:")

    #init()
    model = AdalineModel(0,0,0,0,0,True,0)
    init_adaline_model(model)
    #train_model()
    unipolar = False
    training_samples = get_samples(unipolar)
    train_model_adaline(model, training_samples, iterations_limit=100)
    print("w0 = " + str(model.w0) + " w1 = " + str(model.w1) + " w2 = " + str(model.w2))
    print(str(predict(-0.99, -0.99, model.w0, model.w1, model.w2, model.threshold, unipolar)))
    print(str(predict(-1,1, model.w0, model.w1, model.w2, model.threshold, unipolar)))
    print(str(predict(1,-1, model.w0, model.w1, model.w2, model.threshold, unipolar)))
    print(str(predict(0.99,0.99,model.w0,model.w1,model.w2, model.threshold, unipolar)))

    '''
    model_perceptron = PerceptronModel(0,0,0,1, False, 0.3)
    init_perceptron_model(model_perceptron)
    unipolar = True
    training_samples = get_samples(unipolar)
    train_model(model_perceptron,training_samples, isUnipolar=unipolar)

    model_adaline = AdalineModel(0,0,0,1,0, False, 0)
    init_adaline_model(model_adaline)
    unipolar = False
    training_samples = get_samples(unipolar)
    train_model_adaline(model_adaline, training_samples)

'''

__main__()