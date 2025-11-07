import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn

from utils.data_loader_phasor import MyDataLoader
from utils.loss import JointLoss, MSEAADLoss
from model.UNet import UNet
from sklearn.cluster import KMeans
from model.Clustering import clustering_loss
from scipy.optimize import linear_sum_assignment

parser = argparse.ArgumentParser()

# model settings
parser.add_argument("--loss", type=str, default="MSELoss")
parser.add_argument("--model", default="UNet", type=str)

# training settings
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--load_init_model_iter", type=int, default=350) # initial loading weights
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--test_interval", type=int, default=10000)
parser.add_argument("--valid_interval", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)

# lr decay
parser.add_argument("--start_lr", type=float, default=0.0002)
parser.add_argument("--lr_decay_factor", type=float, default=0.5)

# about data
parser.add_argument("--data_dir", type=str, default="/data/wangyq/MyHU/synthetic_data_128")
parser.add_argument("--dataset_name", type=str, default="synthetic")
parser.add_argument("--save_weights_dir", type=str, default='saved_models/')
parser.add_argument("--save_weights_suffix", type=str, default="_phasor_clustering_128")

# about image
parser.add_argument("--image_size", type=int, default=128)
parser.add_argument("--channels", type=int, default=5)
parser.add_argument("--fluorophore_num", type=int, default=4)
parser.add_argument("--autofluorophore", type=int, default=1)

args = parser.parse_args()

load_init_model_iter = args.load_init_model_iter
epochs = args.epochs
test_interval = args.test_interval
valid_interval = args.valid_interval
batch_size = args.batch_size

loss = args.loss
model = args.model
start_lr = args.start_lr
lr_decay_factor = args.lr_decay_factor

data_dir = args.data_dir
dataset_name = args.dataset_name
save_weights_dir = args.save_weights_dir
save_weights_suffix = args.save_weights_suffix

image_size = args.image_size
channels = args.channels
fluorophore_num = args.fluorophore_num
autofluorophore = args.autofluorophore

cuda = torch.cuda.is_available()
torch.cuda.set_device(args.gpu_id)

# define and make paths
save_weights_folder = dataset_name + '_' + model + save_weights_suffix
save_weights_path = save_weights_dir + save_weights_folder + '/'
data_images_path = data_dir + '/mixed_image'
data_gt_path = data_dir + '/unmixed_image'
if not os.path.exists(save_weights_path):
    os.makedirs(save_weights_path)
with open(save_weights_path + "config.txt","a") as f:
    f.write('\n\n')
    f.write(str(args))

torch.autograd.set_detect_anomaly(True)

# --------------------------------------------------------------------------------
#                        calculate and process phasor plot
# --------------------------------------------------------------------------------
# load images and convert the input image into phase plot coordinates
transform = transforms.Compose([
    transforms.ToTensor(),
])
full_dataset = MyDataLoader(data_images_path, data_gt_path, transform=transform)
print(f"Dataset size: {len(full_dataset)}\n")

# split dataset
train_size = int(0.8 * len(full_dataset))
#test_size = int(0.2 * len(full_dataset))
val_size = len(full_dataset) - train_size

#train_dataset, test_dataset, val_dataset = random_split(full_dataset, [train_size, test_size, val_size])
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
#test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# --------------------------------------------------------------------------------
#                           select models and optimizer
# --------------------------------------------------------------------------------
n_classes = fluorophore_num + autofluorophore
model = UNet(n_channels=2, n_classes=2)
#model = VAE(input_size=2*image_size*image_size, hidden_size=512, latent_size=50)
kmeans = KMeans(n_clusters=n_classes)

optimizer = optim.Adam(model.parameters(), lr=start_lr)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: start_lr * (lr_decay_factor ** (step / epochs)))

if loss == "MSELoss":
    loss_fn = nn.MSELoss()
elif loss == "MSE_AADLoss":
    loss_fn = MSEAADLoss(lambda_space=1.0, lambda_freq=0.5)
elif loss == "JointLoss":
    loss_fn = JointLoss(initial_alpha=0.1, final_alpha=0.01, total_steps=epochs)

# load existing weight
if os.path.exists(save_weights_path + 'weights_' + str(load_init_model_iter) + '.pth'):
    model.load_state_dict(torch.load(save_weights_path + 'weights_' + str(load_init_model_iter) + '.pth'))
    print('Loading weights successfully: ' + save_weights_path + 'weights_' + str(load_init_model_iter) + '.pth')


# --------------------------------------------------------------------------------
#                           training and validation
# --------------------------------------------------------------------------------
best_val_loss = float('inf')
train_losses = []
train_losses_recon = []
train_losses_gt = []
valid_losses = []
test_losses = []
if cuda:
    model.cuda()

for epoch in range(epochs - load_init_model_iter):
    model.train()
    running_loss_recon = 0
    running_loss_gt = 0
    running_loss = 0

    for data, target in train_loader:
        if cuda:
            data = data.cuda()  # (B, 2, H, W)
            target = target.cuda()

        optimizer.zero_grad()
        #flattened_data = torch.flatten(data, start_dim=1)
        #model_output, mu, logvar = model(data)
        #loss_recon = model.loss_function(model_output, data, mu, logvar)
        
        # adjust phasor plot
        model_output = model(data)
        #loss_recon = loss_fn(model_output, data)

        # Clustering
        clustering_output = []
        reordered_output = []
        loss_c = 0
        train_batch_size = data.size(0)

        for pp in model_output:
            pp = pp.view(2, image_size*image_size).permute(1, 0)  # (H*W, 2)
        
            if isinstance(pp, torch.Tensor):
                pp_numpy = pp.cpu().detach().numpy()

            labels = kmeans.fit_predict(pp_numpy)
            cluster_centers = kmeans.cluster_centers_
            if cuda:
                cluster_centers = torch.tensor(cluster_centers, dtype=torch.float32).cuda()
                labels_tensor = torch.tensor(labels, dtype=torch.long).cuda()

            #loss_c += clustering_loss(pp, labels_tensor, cluster_centers)
            #save_splited_image(pp, labels, 1)

            # Reconstruct the unmixing result
            labels_image = labels.reshape(image_size, image_size)
            labels_image = torch.tensor(labels_image, dtype=torch.long)
            # Use one_hot to generate one-hot encoding
            one_hot = torch.nn.functional.one_hot(labels_image, num_classes=n_classes)  # (H, W, num_clusters)
            one_hot = one_hot.permute(2, 0, 1).float()  # (num_clusters, H, W)

            # Convert output to NumPy array
            output_np = one_hot.squeeze().cpu().numpy()
            # Ensure data is in range 0-1
            output_np = np.clip(output_np, 0, 1).astype(np.float32)
            clustering_output.append(output_np)
        
        # reorder output
        target_np = target.squeeze().cpu().numpy()
        # Ensure data is in range 0-1
        target_np = np.clip(target_np, 0, 1).astype(np.float32)
        for i in range(train_batch_size):
            new_image = reorder_image_channels(target_np[i], clustering_output[i])
            reordered_output.append(new_image)
        
        reordered_output = np.array(reordered_output)
        reordered_output = torch.from_numpy(reordered_output).float()
        reordered_output.requires_grad_(True)
        if cuda:
            reordered_output = reordered_output.cuda()
        
        loss_gt = loss_fn(reordered_output, target)
        loss = loss_gt

        loss.backward()
        optimizer.step()

        #running_loss_recon += loss_recon.item()
        running_loss_gt += loss_gt.item()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    #train_losses_recon.append(running_loss_recon / len(train_loader))
    train_losses_gt.append(running_loss_gt / len(train_loader))
    print(f'Epoch [{epoch + 1 + load_init_model_iter}/{epochs}], '
      f'Train Loss: {epoch_loss:.8f}, ')
      #f'Recon Loss: {running_loss_recon / len(train_loader):.8f}, '
      #f'GT Loss: {running_loss_gt / len(train_loader):.8f}')

    if ((epoch + 1 + load_init_model_iter) % valid_interval == 0):
        model.eval()
        val_loss_recon = 0
        val_loss_gt = 0
        val_loss = 0
        with torch.no_grad():
            for data, target in valid_loader:
                if cuda:
                    data = data.cuda()
                    target = target.cuda()

                #flattened_data = torch.flatten(data, start_dim=1)
                # model_output, mu, logvar = model(data)
                # loss_recon = model.loss_function(model_output, data, mu, logvar)
                model_output = model(data)
                #loss_recon = loss_fn(model_output, data)

                # Clustering
                clustering_output = []
                reordered_output = []
                loss_c = 0
                val_batch_size = data.size(0)

                for pp in model_output:
                    pp = pp.view(2, image_size*image_size).permute(1, 0)  # (H*W, 2)
                
                    if isinstance(pp, torch.Tensor):
                        pp_numpy = pp.cpu().detach().numpy()

                    labels = kmeans.fit_predict(pp_numpy)
                    cluster_centers = kmeans.cluster_centers_
                    if cuda:
                        cluster_centers = torch.tensor(cluster_centers, dtype=torch.float32).cuda()
                        labels_tensor = torch.tensor(labels, dtype=torch.long).cuda()

                    #loss_c += clustering_loss(pp, labels_tensor, cluster_centers)
                    #save_splited_image(pp, labels, 1)

                    # Reconstruct the unmixing result
                    labels_image = labels.reshape(image_size, image_size)
                    labels_image = torch.tensor(labels_image, dtype=torch.long)
                    # Use one_hot to generate one-hot encoding
                    one_hot = torch.nn.functional.one_hot(labels_image, num_classes=n_classes)  # (H, W, num_clusters)
                    one_hot = one_hot.permute(2, 0, 1).float()  # (num_clusters, H, W)
                    
                    # Convert output to NumPy array
                    output_np = one_hot.squeeze().cpu().numpy()
                    # Ensure data is in range 0-1
                    output_np = np.clip(output_np, 0, 1).astype(np.float32)
                    clustering_output.append(output_np)
                
                # reorder output
                target_np = target.squeeze().cpu().numpy()
                # Ensure data is in range 0-1
                target_np = np.clip(target_np, 0, 1).astype(np.float32)
                for i in range(val_batch_size):
                    new_image = reorder_image_channels(target_np[i], clustering_output[i])
                    reordered_output.append(new_image)
                
                reordered_output = np.array(reordered_output)
                reordered_output = torch.from_numpy(reordered_output).float()
                reordered_output.requires_grad_(True)
                if cuda:
                    reordered_output = reordered_output.cuda()
                
                loss_gt = loss_fn(reordered_output, target)
                val_loss_gt += loss_gt.item()
                #val_loss_recon += loss_recon.item()
                val_loss = val_loss_gt
    
        val_loss /= len(valid_loader)
        valid_losses.append(val_loss)
        torch.save(model.state_dict(), save_weights_path+'weights_'+str(epoch+1+load_init_model_iter)+'.pth')
        print(f'Validation Loss: {val_loss:.8f}, ')
            #f'Recon Loss: {val_loss_recon / len(valid_loader):.8f}, '
            #f'GT Loss: {val_loss_gt / len(valid_loader):.8f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_weights_path + 'best_model.pth')
            print(f'Best model updated at epoch {epoch+1+load_init_model_iter}')

    # if (epoch + 1 + load_init_model_iter) % test_interval == 0:
    #     model.load_state_dict(torch.load(save_weights_path + 'best_model.pth'))
    #     print(f'Loading best model successfully')
    #     model.eval()
    #     test_loss = 0
    #     with torch.no_grad():
    #         for data, target in test_loader:
    #             if cuda:
    #                 data = data.cuda()
    #                 target = target.cuda()
    #             output = model(data)
    #             test_loss += loss_fn(output, target).item()

    #     test_loss /= len(test_loader)
    #     test_losses.append(test_loss)
    #     print(f'Test Loss: {test_loss:.8f}')

    lr_scheduler.step() # update learning rate


# --------------------------------------------------------------------------------
#                                 plot diagram
# --------------------------------------------------------------------------------
# Generate the epoch list corresponding to the x axis
total_epochs = epochs
epochs = list(range(load_init_model_iter + 1, total_epochs + 1))
valid_epochs = list(range(load_init_model_iter + valid_interval, total_epochs + 1, valid_interval))
#test_epochs = list(range(test_interval, len(train_losses) + 1, test_interval))

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Training Loss', marker='o')
#plt.plot(epochs, train_losses_recon, label='Training Recon Loss', marker='o')
#plt.plot(epochs, train_losses_gt, label='Training GT Loss', marker='o')
plt.plot(valid_epochs, valid_losses, label='Validation Loss', marker='o')
#plt.plot(test_epochs, test_losses, label='Test Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Training And Validation Loss Over Epochs')
plt.legend()
plt.grid(True)

plt.savefig(save_weights_path+'loss_plot.png', dpi=300, bbox_inches='tight')
plt.show()
