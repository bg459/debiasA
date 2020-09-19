
import argparse
import sys


from src.functions import *


# Runs all the functions necessary. Returns if any of the intermediate steps fails

def runner():
    print("### TRAINING THE BASE MODEL###")
    #naked_model("base_model.h5", "processed_train.pkl", "train_labels.pkl", "processed_test.pkl", "test_labels.pkl")
    combined_model("base_model.h5", "processed_train.pkl", "train_labels.pkl", "processed_test.pkl", "test_labels.pkl", "race", 0.05)
# Parse arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='debiasA')
    parser.add_argument('--base', type = str, help = "The file location (.h5) of the base neural network", default = "base_model.h5")
    parser.add_argument('--train', type = str, help = "The file location (.pkl) of processed training set", default = "processed_train.pkl")
    parser.add_argument('--test', type = str, help = "The file location (.pkl) of processed test set", default = "processed_test.pkl")
    parser.add_argument('--alpha', type = float, help = "Measure of the adversarial \
    correction for resulting network. Larger 0 < alpha < 1 is more extreme")
    #args = parser.parse_args()

    runner()
