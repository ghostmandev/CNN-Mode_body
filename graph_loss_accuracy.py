import plotly
import plotly.graph_objs as go

class Graph():
    def loss_accuracy(Our_model):
        # Plot Loss
        h1 = go.Scatter(y=Our_model.history['loss'], mode='lines', line=dict(width=2, color='blue'), name='loss')

        h2 = go.Scatter(y=Our_model.history['val_loss'], mode='lines', line=dict(width=2, color='red'), name='val_loss')

        data = [h1, h2]

        layout1 = go.Layout(title='Loss', xaxis=dict(title='epochs'), yaxis=dict(title=''))
        fig1 = go.Figure(data=data, layout=layout1)
        plotly.offline.iplot(fig1)

        # print(Our_model.history.keys())

        # Plot Accuracy
        h1 = go.Scatter(y=Our_model.history['accuracy'], mode='lines', line=dict(width=2, color='blue'), name='acc')

        h2 = go.Scatter(y=Our_model.history['val_accuracy'], mode='lines', line=dict(width=2, color='red'),
                        name='val_acc')

        data = [h1, h2]

        layout1 = go.Layout(title='Accuracy', xaxis=dict(title='epochs'), yaxis=dict(title=''))
        fig1 = go.Figure(data=data, layout=layout1)
        plotly.offline.iplot(fig1)