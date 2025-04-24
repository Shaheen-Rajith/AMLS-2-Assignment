import os
import torch
import torch.nn as nn
import gc

from A.spec_gen import generate_spectrograms
from A.data_preproc import data_preprocessing

from A.trans_model import Trans_CNN
from A.cnn_model import Self_CNN
from A.utils import train_model, training_plots, test_model, confusion_mat

# ======================================================================================================================
# Spectrogram Image Generation
spectrogram_path = "Datasets/Spec-Images"
if not os.path.exists(spectrogram_path):
    generate_spectrograms(
        input_dir = "Datasets/Sound-Files",
        output_dir = spectrogram_path,
        sample_rate = 32000,
        duration = 5,
        n_mels = 128
    )
else:
    print("Spectrogram Images Already Exist")

# ======================================================================================================================
# Data preprocessing
class_names, train_loader, val_loader, test_loader = data_preprocessing(
    spec_dir = spectrogram_path,
    train_ratio = 0.7,
    seed = 37
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================================================================================
# Task A - Selfmade CNN Based Model
model_A_self = Self_CNN(num_classes= 24)  # Build model object.

model_A_self, self_train_losses, self_val_losses = train_model(
    model = model_A_self,
    train_loader = train_loader,
    val_loader = val_loader,
    n_epochs = 50,
    lr = 6e-4,
    device = device,
    criterion = nn.CrossEntropyLoss()
)

training_plots(
    train_losses = self_train_losses,
    val_losses = self_val_losses,
    n_epochs = 50,
    title = "Self_Loss"
)

acc_A_train_self, _ = test_model( # Calculate Final Training Loss
    model = model_A_self, 
    data_loader = train_loader, 
    device = device
)

acc_A_test_self, cm_self = test_model( # Calculate Final Testing Loss
    model = model_A_self, 
    data_loader = test_loader, 
    device = device
)

confusion_mat(
    cm = cm_self,
    class_names = class_names,
    title = "Self_CM" 
)


gc.collect()
torch.cuda.empty_cache()


# ======================================================================================================================
# Task A - Transfer Learning Based Model
model_A_trans = Trans_CNN(num_classes= 24)  # Build model object.

model_A_trans, trans_train_losses, trans_val_losses = train_model(
    model = model_A_trans,
    train_loader = train_loader,
    val_loader = val_loader,
    n_epochs = 4,
    lr = 5e-5,
    device = device,
    criterion = nn.CrossEntropyLoss()
)

training_plots(
    train_losses = trans_train_losses,
    val_losses = trans_val_losses,
    n_epochs = 4,
    title = "Trans_Loss"
)

acc_A_train_trans, _ = test_model( # Calculate Final Training Loss
    model = model_A_trans, 
    data_loader = train_loader, 
    device = device
)

acc_A_test_trans, cm_trans = test_model( # Calculate Final Testing Loss
    model = model_A_trans, 
    data_loader = test_loader, 
    device = device
)

confusion_mat(
    cm = cm_trans,
    class_names = class_names,
    title = "Trans_CM" 
)


gc.collect()
torch.cuda.empty_cache()




# ======================================================================================================================
## Print out your results with following format:
print('TA (Custom CNN):{:.5f},{:.5f}; TA (Transfer Learning):{:.5f},{:.5f};'.format(acc_A_train_self, acc_A_test_self,
                                                        acc_A_train_trans, acc_A_test_trans))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A_train = 'TBD'
# acc_B_test = 'TBD'