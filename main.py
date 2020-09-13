
import argparse




# Parse arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='debiasA')
    parser.add_argument('--base', type = str, help = "The file location (.h5) of the base neural network", default = "base_model.h5")
    parser.add_argument('--train', type = str, help = "The file location (.pkl) of processed training set", default = "processed_train.pkl")
    parser.add_argument('--test', type = str, help = "The file location (.pkl) of processed test set", default = "processed_test.pkl")
    parser.add_argument('--alpha', type = float, help = "Measure of the adversarial \
    correction for resulting network. Larger 0 < alpha < 1 is more extreme")
    args = parser.parse_args()
