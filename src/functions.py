## Functions to be imported in main.py
from tensorflow import keras

# Run the model naked, i.e. with no corrections. If this fails then there is
# an issue with the inputs. Also want to see what the bias is like here.
def naked_model(model_path, train_path, feature):
    # Loading the simple model.
    model = keras.models.load_model(model_path)
    
