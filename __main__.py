from model.validator import validate_model
from model.data import load_data

if __name__ == '__main__':
    print('Loading data ...')
    X, y = load_data()

    print('Loaded.')
    best_thres = validate_model(X, y)
