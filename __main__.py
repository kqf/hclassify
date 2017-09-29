from model.validator import validate_model
from model.data import DataHandler as dhlr

if __name__ == '__main__':
    print('Loading data ...')
    X, y = dhlr.load_data()

    print('Loaded.')
    best_thres = validate_model(X, y)
