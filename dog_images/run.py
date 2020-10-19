import torch.optim as optim
import DogBreedClassifier
from path_names import train_directory, test_directory, val_directory, model_path

visualize = False
train_nw = False
batch = 20
epochs = 100
device = DogBreedClassifier.get_device()
train_loader, test_loader, val_loader = DogBreedClassifier.load_data(train_directory, test_directory,
                                                                     val_directory, batch_size=batch)
if visualize:
    DogBreedClassifier.visualize_data(train_loader)
nw_model = DogBreedClassifier.DogBreedNet()
train_device = DogBreedClassifier.get_device()
loss_criterion = DogBreedClassifier.nn.NLLLoss()
optimizer_loss = optim.SGD(nw_model.parameters(), momentum=0.9, lr=0.009)
if train_nw:
    training_loss, validation_loss, validation_accuracy = \
        DogBreedClassifier.train_network(train_loader, val_loader, nw_model, train_device, loss_criterion,
                                         optimizer_loss, epochs=epochs)
optimized_model = DogBreedClassifier.load_model(model_path)
# call test function
if optimized_model is not None:
    DogBreedClassifier.test(test_loader, optimized_model, loss_criterion, device)
