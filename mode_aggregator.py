import pickle
from statistics import mode
from sklearn.metrics import accuracy_score
import numpy as np

class ModeAggregator():
    def ModeFunction(model, target_labels):

        print('=================Start Mode Aggregator=================')
        # load data for test
        data_bkT = open('D:\\TSLT\\PyCharm\\ImageClassify\\backupData\\dataTest_image_2828.pkl', 'rb')
        data_test = pickle.load(data_bkT)
        data_bkT.close()

        # load label for test
        labels_bkT = open('D:\\TSLT\\PyCharm\\ImageClassify\\backupData\\labelsTest_image_2828.pkl', 'rb')
        labels_test = pickle.load(labels_bkT)
        labels_bkT.close()

        print(data_test.shape, labels_test.shape)

        result = model.predict(data_test)
        result = np.argmax(result, axis=1)

        predict = list()
        for index in result:
            pred = target_labels[index]
            predict = np.append(predict, pred)
        print(predict)
        print(len(predict))

        index = int(len(result) / 75)
        start = 0
        end = 75
        for i in range(1, index + 1):
            print("Prediction = %s" % (mode(predict[start:end * i])) + '; True = ', (mode(labels_test[start:end * i])))
            start += 75

        print('Accuracy (Mode Aggregator) = ', accuracy_score(predict, labels_test) * 100)