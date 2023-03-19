from args_parser import ArgumentParser
from config import Config
from dataset import CTRDataset
from torch.utils.data import DataLoader
from model import Model
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from sklearn import metrics
from sklearn.metrics import log_loss
from datetime import datetime
from snntorch import surrogate
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
def print_batch_accuracy(net, config, data, targets, train=False):
    output, _ = net(data.view(config.batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.5f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer(
    net, config, data, targets, epoch,
    counter, iter_counter,
        loss_hist, test_loss_hist, test_data, test_targets):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set MSE Loss: {loss_hist[counter]:.5f}")
    print(f"Test Set MSE Loss: {test_loss_hist[counter]:.5f}")
    print_batch_accuracy(net, config, data, targets, train=True)
    print_batch_accuracy(net, config, test_data, test_targets, train=False)
    print("\n")

def run():
    
    # Parse arguments
    args = ArgumentParser.parse()

    # Configurations
    config = Config(args)

    best_auc = 0
    time_best_auc = 0
    
    # Load train/test dataset
    #train_dataset = CTRDataset(csv_file='.tmp/sample_train_criteo_dataset.csv')
    #test_dataset = CTRDataset(csv_file='.tmp/sample_test_criteo_dataset.csv')

    if config.dataset == "avazu":
        config.num_inputs = 22
        train_dataset = CTRDataset(csv_file='.tmp/sample_train_avazu_dataset.csv')
        test_dataset = CTRDataset(csv_file='.tmp/sample_test_avazu_dataset.csv')
    if config.dataset == "custom":
        df = pd.read_csv(".tmp/dataset_train.csv")
        config.num_inputs = len(df.columns) - 1
        train_dataset = CTRDataset(csv_file=".tmp/dataset_train.csv")
        test_dataset = CTRDataset(csv_file=".tmp/dataset_test.csv")

    print("Dataset: ",config.dataset)

    # Pass datasets to data loaders 
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    # Setup device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create model
    net = Model(config).to(device)
    
    # Define loss and optimaztion
    #loss = nn.BCEWithLogitsLoss()
    ##loss = nn.BCEWithLogitsLoss()
    #loss = nn.MSELoss()
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)

    loss_hist = []
    test_loss_hist = []
    counter = 0

    now = datetime.now()
    for epoch in range(config.num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)
        log_loss_hist = []
        auc_hist = []

        # Minibatch training loop
        for data, targets in train_batch:
            data = data.to(device)
            targets = targets.to(device)
            
            # forward pass
            net.train()
            spk_rec, mem_rec = net(data.view(config.batch_size, -1))    
           
            # loss / sum over time
            loss_val = torch.zeros((1), dtype=torch.float, device=device)
            for step in range(config.num_steps):
                loss_val += loss(mem_rec[step], targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for plotting
            loss_hist.append(loss_val.item())
            
            # Test set
            with torch.no_grad():
                net.eval()
                test_data, test_targets = next(iter(test_loader))
                test_data = test_data.to(device)
                test_targets = test_targets.to(device)

                # Test set forward pass
                test_spk, test_mem = net(test_data.view(config.batch_size, -1))

                # Test set loss
                test_loss = torch.zeros((1), dtype=torch.float, device=device)
                for step in range(config.num_steps):
                    test_loss += loss(test_mem[step], test_targets)
                    logl = log_loss(test_targets.cpu().detach().numpy(), test_mem[step].cpu().detach().numpy(), eps=1e-15)
                    log_loss_hist.append(logl)
                    auc = metrics.roc_auc_score(test_targets.cpu().detach().numpy(), test_mem[step].cpu().detach().numpy())
                    auc_hist.append(auc)

                test_loss_hist.append(test_loss.item())
                
                
                # Print train/test loss/accuracy
                if counter % 50 == 0:
                    print("\n-------------------------------------------------")
                    currentLogLoss = np.mean(log_loss_hist)
                    currentAUC = np.mean(auc_hist)
                    print("Epochs: ", str((epoch + 1)) + "/" + str(config.num_epochs))
                    print("Logloss: " + str(currentLogLoss))
                    print("AUC: " + str(currentAUC))
                    later = datetime.now()
                    difference = (later - now).total_seconds()

                    if currentAUC > best_auc:
                        best_auc = currentAUC
                        time_best_auc = difference

                    print("AUC(%): " + "{:.2%}".format(currentAUC))
                    pytorch_total_params = sum(p.numel() for p in net.parameters())
                    print('Total params: %d' % pytorch_total_params)
                    print('Time: %d s' % (difference))
                    if best_auc > 0:
                        print("-------------------------------------------------")
                        print("Best AUC: " +str(best_auc))
                        print("Best AUC(%): " + "{:.2%}".format(best_auc))
                        print("Best AUC(Time): %d s" % time_best_auc)
                        print("-------------------------------------------------\n")
                    else:
                        print("-------------------------------------------------\n")
                    #train_printer(
                       #net, config, data, targets, epoch,
                        #counter, iter_counter,
                        #loss_hist, test_loss_hist,
                        #test_data, test_targets)
                counter += 1
                iter_counter += 1
    # Plot Loss
    if config.plot:
        fig = plt.figure(facecolor="w", figsize=(10, 5))
        plt.plot(loss_hist)
        plt.plot(test_loss_hist)
        plt.title("Loss Curves")
        plt.legend(["Train Loss", "Test Loss"])
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()

if __name__ == "__main__":
    run()
