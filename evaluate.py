import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score
import plotly.figure_factory as ff

class Evaluation():
    def val_test(model, train_gen, x_test, y_test):
        ###################### Loss & Accuracy (train and validation)######################################

        loss, acc = model.evaluate(train_gen, verbose=1)
        print('Loss: %f, Accuracy: %f ' % (loss, acc * 100))

        ####################### Evaluattion (y_test) ##############################################

        y_test_class = pd.get_dummies(y_test).values
        print(y_test_class.shape)

        value = model.predict(x_test)
        y_pred = np.argmax(value, axis=1)
        y_true = np.argmax(y_test_class, axis=1)

        print(y_true.shape, y_pred.shape)

        print('Accuracy = ', accuracy_score(y_true, y_pred))
        print('Precission = ', np.average(precision_score(y_true, y_pred, average='weighted')))
        print('Recall = ', recall_score(y_true, y_pred, average='weighted'))
        print('F1 Score = ', f1_score(y_true, y_pred, average='macro'))

        # print('Confusion Matrix :', confusion_matrix(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

        target_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        print(classification_report(y_true, y_pred, target_names=target_labels))

    def cm_plot(cm, labels):
        x = labels
        y = labels

        z_text = [[str(y) for y in x] for x in cm]
        fig = ff.create_annotated_heatmap(cm, x=x, y=y, annotation_text=z_text, colorscale='blues')

        fig.update_layout(title_text='Confusion Matrix')
        fig.add_annotation(
            dict(font=dict(color='black', size=13), x=0.5, y=-0.15, showarrow=False, text='Predicted Value',
                 xref='paper',
                 yref='paper'))
        fig.add_annotation(
            dict(font=dict(color='black', size=13), x=-0.2, y=0.5, showarrow=False, text='Real Value', textangle=-90,
                 xref='paper', yref='paper'))
        fig.update_layout(margin=dict(t=50, l=200))
        fig['layout']['yaxis']['autorange'] = 'reversed'
        fig['data'][0]['showscale'] = True
        fig.show()
