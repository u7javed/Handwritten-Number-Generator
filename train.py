import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import time
import matplotlib.pyplot as plt
import argparse
import torch.optim as optim
import os

from models import *

class Trainer():

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def __init__(self, data_directory, embedding_dim, batch_size, latent_size=100, device='cpu', lr=0.0002, num_workers=1):
        #load dataset
        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])

        #check if data directory exists, if not, create it.
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
            print('Directory created.')
        else:
            print('Directory exists.')

        #get dataset from directory. If not present, download to directory
        self.dataset = torchvision.datasets.MNIST(data_directory, train=True, transform=transformation, download=True)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.batch_size = batch_size
        self.device = device

        self.class_size = 10

        #define models
        self.latent_size = latent_size

        self.dis = Discriminator(self.class_size).to(device)
        self.dis.apply(self.weights_init)
        self.gen = Generator(self.class_size, embedding_dim, latent_size).to(device)
        self.gen.apply(self.weights_init)

        self.loss_func = nn.BCELoss().to(device)

        self.optimizer_d = optim.Adam(self.dis.parameters(), lr=lr)
        self.optimizer_g = optim.Adam(self.gen.parameters(), lr=lr)

    def train(self, epochs, saved_image_directory, saved_model_directory):
        #train
        epochs = 200

        REAL_LABEL = 1
        FAKE_LABEL = 0

        start_time = time.time()

        for epoch in range(epochs):
            elapsed_time = time.time()
            gen_loss = 0
            dis_loss = 0
            for images, labels in self.data_loader:

                for d in self.dis.parameters():
                    d.data.clamp_(-0.01, 0.01)

                b_size = len(images)
                self.optimizer_d.zero_grad()
                #Real Loss on Discriminator
                real_labels = torch.full((b_size,), REAL_LABEL).to(self.device)
                pred_real = self.dis(images.to(self.device), labels.to(self.device))
                d_loss_real = self.loss_func(pred_real, real_labels)

                #Fake Loss on Discriminator
                z = torch.randn(b_size, self.latent_size).to(self.device)
                rand_labels = torch.LongTensor(np.random.randint(0, self.class_size, b_size)).to(self.device)
                fake_images = self.gen(z, rand_labels)
                fake_labels = torch.full((b_size,), FAKE_LABEL).to(self.device)
                pred_fake = self.dis(fake_images, rand_labels)
                d_loss_fake = self.loss_func(pred_fake, fake_labels)

                d_loss = d_loss_fake + d_loss_real
                d_loss.backward()
                self.optimizer_d.step()

                dis_loss += d_loss.item()/len(images)

                #Train Generator
                self.optimizer_g.zero_grad()
                z = torch.randn(b_size, self.latent_size).to(self.device)
                rand_labels = torch.LongTensor(np.random.randint(0, self.class_size, b_size)).to(self.device)
                fake_images = self.gen(z, rand_labels)
                pred_fake = self.dis(fake_images, rand_labels)
                g_loss = self.loss_func(pred_fake, real_labels)
                g_loss.backward()
                self.optimizer_g.step()

                gen_loss += g_loss.item()/len(images)

            #gen_labels = torch.LongTensor([i for i in range(10)]).to('cuda')
            gen_labels = torch.LongTensor([i for i in range(10)]).view(-1).to(self.device)
            z = torch.randn(len(gen_labels), self.latent_size).to(self.device)
            gen_images = self.gen(z, gen_labels)

            grid_img = torchvision.utils.make_grid(gen_images.cpu().detach(), nrow=5, normalize=True)
            _, ax = plt.subplots(figsize=(12, 12))
            plt.axis('off')
            ax.imshow(grid_img.permute(1, 2, 0)) 
            plt.savefig(saved_image_directory + '/checkpoint_image_{}.png'.format(epoch), bbox_inches='tight')

            print('Epoch {},    gen_loss: {:.4f},   dis_loss: {:.4f}'.format(
                epoch,
                gen_loss,
                dis_loss
            ))
    
            if epoch % 5 == 0:
                torch.save(self.gen.state_dict(), saved_model_directory + '/generator_{}.pt'.format(epoch))
                torch.save(self.dis.state_dict(), saved_model_directory + '/discriminator_{}.pt'.format(epoch))

            elapsed_time = time.time() - elapsed_time

            print('Taken {:.4f} seconds to finish epoch. Estimated {:.4f} hours remaining'.format(
                elapsed_time,
                (elapsed_time*(epochs - epoch))/(3600)
            ))
        final_elapsed_time = time.time() - start_time
        print('Training Finished. Took {:.4f} hours or {:.4f} seconds'.format(final_elapsed_time))
        return final_elapsed_time

def main():
    parser = argparse.ArgumentParser(description='Hyperparameters for training GAN')
    #hyperparameter loading
    parser.add_argument('--data_directory', type=str, default='data', help='directory to MNIST dataset files')
    parser.add_argument('--saved_image_directory', type=str, default='data/saved_images', help='directory to where image samples will be saved')
    parser.add_argument('--saved_model_directory', type=str, default='saved_models', help='directory to where model weights will be saved')
    parser.add_argument('--embedding_dim', type=int, default=10, help='size of embedding vector')
    parser.add_argument('--batch_size', type=int, default=64, help='size of batches passed through networks at each step')
    parser.add_argument('--latent_size', type=int, default=100, help='size of gaussian noise vector')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or gpu depending on availability and compatability')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate of models')
    parser.add_argument('--num_workers', type=int, default=0, help='workers simultaneously putting data into RAM')
    parser.add_argument('--epochs', type=int, default=100, help='number of iterations of dataset through network for training')
    args = parser.parse_args()

    data_dir = args.data_directory
    saved_image_dir = args.saved_image_directory
    saved_model_dir = args.saved_model_directory
    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    latent_size = args.latent_size
    device = args.device
    lr = args.lr
    num_workers = args.num_workers
    epochs = args.epochs

    gan = Trainer(data_dir, embedding_dim, batch_size, latent_size, device, lr, num_workers)
    train_time = gan.train(epochs, saved_image_dir, saved_model_dir)


if __name__ == "__main__":
    main()
