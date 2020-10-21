import torch.optim as optima
import dog_breed_vgg16
from path_names import train_directory, test_directory, val_directory, model_path
import params

lr = 0.003
device, _ = dog_breed_vgg16.get_device()
nw_model_transfer = dog_breed_vgg16.get_transfer_model()
criterion_transfer = dog_breed_vgg16.nn.CrossEntropyLoss()
optimizer_transfer = optima.SGD(nw_model_transfer.classifier.parameters(), lr=lr)
train_device, use_cuda = dog_breed_vgg16.get_device()
train_loader, test_loader, val_loader = dog_breed_vgg16.load_data_vgg(train_directory, test_directory,
                                                                      val_directory, batch_size=params.batch_size,
                                                                      error_check=params.error_check,
                                                                      input_size=params.input_size)
# train the model
if params.train_transfer_model:
    training_loss_transfer, validation_loss_transfer, validation_accuracy_transfer = \
        dog_breed_vgg16.train_network_vgg(train_loader, val_loader, nw_model_transfer, train_device,
                                          criterion_transfer, optimizer_transfer,
                                          epochs=params.epochs)
transfer_model1 = dog_breed_vgg16.get_transfer_model()
transfer_model1 = dog_breed_vgg16.load_transfer_model(model_path, transfer_model1)
if transfer_model1 is not None:
    dog_breed_vgg16.test(test_loader, transfer_model1, criterion_transfer, device)
