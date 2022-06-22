import pickle

bk_dataPath = 'D:\\TSLT\\PyCharm\\ImageClassify\\backupData\\cnn_ma_data_200622_1_to_9_2828.pkl'
bk_labelPath = 'D:\\TSLT\\PyCharm\\ImageClassify\\backupData\\cnn_ma_labels_200622_1_to_9_2828.pkl'

class LoadData():
    def LoadBK():
        bk_data = open(bk_dataPath, 'rb')
        bk_label = open(bk_labelPath, 'rb')
        data = pickle.load(bk_data)
        labels = pickle.load(bk_label)
        bk_data.close(), bk_label.close()
        print(data.shape, labels.shape)
        return data, labels
