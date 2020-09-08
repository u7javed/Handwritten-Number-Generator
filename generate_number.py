
import torch
import torchvision

import matplotlib.pyplot as plt
import argparse

from models import Generator

#takes in a number and generates the handwritten image
def show_number(number, inverted, generator, latent_size, save_directory, device):
    number = list(number)
    for i in range(len(number)):
        try:
            number[i] = int(number[i])
        except:
            print('Error! A number was not passed in as input')
        
    labels = torch.LongTensor([i for i in number]).to(device)
    z_noise = torch.randn(len(number), latent_size).to(device)
    generated_images = generator(z_noise, labels)
    #compress images along the x-axis to make these blend into one image
    generated_images = generated_images[:,:,:,5:-5]

    image_grid = torchvision.utils.make_grid(generated_images.cpu().detach(), nrow=len(number), normalize=True, padding=0)
    _, axes = plt.subplots(figsize=(8, 8))
    axes.set_axis_off()
    if inverted is True:
        axes.imshow(1 - image_grid.permute(1, 2, 0), interpolation='nearest') 
        plt.savefig(save_directory + 'generated_number2.png', bbox_inches='tight')
    else:
        axes.imshow(1 - image_grid.permute(1, 2, 0), interpolation='nearest') 
        plt.savefig(save_directory + 'generated_number2.png', bbox_inches='tight')

    print('Image saved to Directory!')

def main():
    #take in arguments
    parser = argparse.ArgumentParser(description='Hyperparameters for training GAN')

    # parameters needed to enhance image
    parser.add_argument('--number', type=str, default='1234', help='number you wish to generate')
    parser.add_argument('--inverted', type=bool, default=False, help='If you want numbers black inverted=False, else True')
    parser.add_argument('--dir_to_generator', type=str, default='best_generator.pt', help='location to ideal generator file')
    parser.add_argument('--class_size', type=int, default=10, help='number of possible single digit numbers')
    parser.add_argument('--embedding_dim', type=int, default=10, help='size of embedding vectors')
    parser.add_argument('--latent_size', type=int, default=100, help='size of latent noise vector')
    parser.add_argument('--save_directory', type=str, default='', help='directory where enhanced image will be saved')

    args = parser.parse_args()

    number = args.number
    inverted = args.inverted
    dir_to_generator = args.dir_to_generator
    class_size = args.class_size
    embedding_dim = args.embedding_dim
    latent_size = args.latent_size
    save_directory = args.save_directory

    #load generator 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    number_generator = Generator(class_size, embedding_dim, latent_size).to(device)
    number_generator.load_state_dict(torch.load(dir_to_generator))

    #generate number
    show_number(number, inverted, number_generator, latent_size, save_directory, device)

if __name__ == "__main__":
    main()