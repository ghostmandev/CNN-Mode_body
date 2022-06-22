
from data_generator import DataGenerator
from load_data import LoadData
from split_data import SplitData
from our_model import Model
from evaluate import Evaluation
from graph_loss_accuracy import Graph
from mode_aggregator import ModeAggregator


################ load data ##############################

data, labels = LoadData.LoadBK()

###############  Train Test Split ##############################

x_train, x_val, x_test, x_val_class, y_val_class, y_test = SplitData.Train_Val_Test_Split(data, labels)

##########################  Model ###############################

model = Model.Our_Model()

model.build()
model.summary()

############################## Model.fit ################################

train_gen = DataGenerator(x_train, x_val_class, 32)
test_gen = DataGenerator(x_val, y_val_class, 32)

My_model = model.fit(train_gen, epochs = 50, verbose = 1, validation_data = test_gen)

#############################  Evaluate and Graph  #####################################

evaluate_model = Evaluation.val_test(model, train_gen, x_test, y_test)

loss_acc_graph = Graph.loss_accuracy(My_model)

##############################  Mode Aggregator  #####################################

target_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
evaluate_mode = ModeAggregator.ModeFunction(model, target_labels)