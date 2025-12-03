import numpy as np
from ai import *

AND_CASES = {
    DataPiece(np.array([0, 0])) : np.array([0]),
    DataPiece(np.array([0, 1])) : np.array([0]),
    DataPiece(np.array([1, 0])) : np.array([0]),
    DataPiece(np.array([1, 1])) : np.array([1])
    }

OR_CASES = {
    DataPiece(np.array([0, 0])) : np.array([0]),
    DataPiece(np.array([0, 1])) : np.array([1]),
    DataPiece(np.array([1, 0])) : np.array([1]),
    DataPiece(np.array([1, 1])) : np.array([1])
    }

XOR_CASES = {
    DataPiece(np.array([0, 0])) : np.array([0]),
    DataPiece(np.array([0, 1])) : np.array([1]),
    DataPiece(np.array([1, 0])) : np.array([1]),
    DataPiece(np.array([1, 1])) : np.array([0])
    }

# (a AND NOT b) XOR (c AND d)
LOGIC_CASES = {}
for a in [0, 1]:
    for b in [0, 1]: 
        for c in [0, 1]:
            for d in [0, 1]:
                LOGIC_CASES[DataPiece(np.array([a, b, c, d]))] = np.array([int((a and not b) ^ (c and d))])

if __name__ == "__main__":
    and_ai = NeuralNetwork([2, 2, 1], [Linear(), ReLU(), Sigmoid()])
    or_ai = NeuralNetwork([2, 2, 1], [Linear(), ReLU(), Sigmoid()])
    # More neurons to accout for harder logic
    xor_ai = NeuralNetwork([2, 4, 1], [Linear(), ReLU(),  Sigmoid()])
    logic_ai = NeuralNetwork([4, 8, 4, 1], [Linear(), ReLU(), ReLU(), Sigmoid()]) 
    
    loss = []

    # Train AND
    and_ai.train(AND_CASES, 1000, 4, 0.9, 0.999)

    # Train OR
    or_ai.train(OR_CASES, 1000, 4, 0.9, 0.999)
    
    # Train XOR
    xor_ai.train(XOR_CASES, 2000, 4, 0.9, 0.999)

    # Train Logic
    logic_ai.train(LOGIC_CASES, 2500, 16, 0.9, 0.999)
    
    # Benchmark AND
    print("----------AND AI Predictions----------")
    for data_input, expected_output  in AND_CASES.items():
        predict = and_ai(data_input.get_data())
        loss.append(abs(expected_output - predict)[0])
        print("For input", data_input.get_data(), "the ai predicted", round(predict[0], 4), "when", expected_output[0], "was expected. Loss", round(loss[-1], 4))
    print("Average loss is", round(sum(loss) / len(loss), 6))

    # Benchmark OR
    print("----------OR AI Predictions----------")
    for data_input, expected_output  in OR_CASES.items():
        predict = or_ai(data_input.get_data())
        loss.append(abs(expected_output - predict)[0])
        print("For input", data_input.get_data(), "the ai predicted", round(predict[0], 4), "when", expected_output[0], "was expected. Loss", round(loss[-1], 4))
    print("Average loss is", round(sum(loss) / len(loss), 6))

    # Benchmark XOR
    print("----------XOR AI Predictions----------")
    for data_input, expected_output  in XOR_CASES.items():
        predict = xor_ai(data_input.get_data())
        loss.append(abs(expected_output - predict)[0])
        print("For input", data_input.get_data(), "the ai predicted", round(predict[0], 4), "when", expected_output[0], "was expected. Loss", round(loss[-1], 4))
    print("Average loss is", round(sum(loss) / len(loss), 6))

    # Benchmark LOGIC
    print("----------LOGIC AI Predictions----------")
    for data_input, expected_output  in LOGIC_CASES.items():
        predict = logic_ai(data_input.get_data())
        loss.append(abs(expected_output - predict)[0])
        print("For input", data_input.get_data(), "the ai predicted", round(predict[0], 4), "when", expected_output[0], "was expected. Loss", round(loss[-1], 4))
    print("Average loss is", round(sum(loss) / len(loss), 6))