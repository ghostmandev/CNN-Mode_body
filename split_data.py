from sklearn.model_selection import train_test_split
import pandas as pd
class SplitData():
    def Train_Val_Test_Split(data, labels):
        # train 60% , test = 20% (val = 20%)
        x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size= 0.6, shuffle = True, random_state = 42)
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        # train 60% , val = 20%, test = 20%
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5, shuffle = True, random_state = 42)
        print('x_train = {}, x_val = {}, x_test = {}'.format(x_train.shape, x_val.shape, x_test.shape))
        print('y_train = {}, y_val = {}, y_test = {}'.format(y_train.shape, y_val.shape, y_test.shape))

        x_val_class = pd.get_dummies(y_train).values
        y_val_class = pd.get_dummies(y_val).values
        print('x_train = {}, x_val = {}, x_test = {}'.format(x_train.shape, x_val.shape, x_test.shape))
        print('y_train = {}, y_val = {}, y_test = {}'.format(y_train.shape, y_val.shape, y_test.shape))
        print('x_val_class = {}, y_val_class = {}'.format(x_val_class.shape, y_val_class.shape))
        print(set(y_val))

        return x_train, x_val, x_test, x_val_class, y_val_class, y_test
