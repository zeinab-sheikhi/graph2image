import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm 
from graph_encoder import ScenceGraphEncoder, UNet
from diffusion_model import SimpleDiffusion


class SceneGraphDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, graph = self.data[idx]
        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        return img, graph


def train():
    
    # Load data 
    dataset = SceneGraphDataset("data/simple_scenes.pt")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gnn = ScenceGraphEncoder().to(device)
    unet = UNet().to(device)
    diffusion = SimpleDiffusion().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        list(gnn.parameters()) + list(unet.parameters()),
        lr=1e-4
    )

    num_epochs = 20
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader)
        for images, graphs in pbar:
            images = images.to(device)

            t = torch.randint(0, diffusion.timesteps, (images.shape[0],))

            noisy_images, noise = diffusion.forward_diffusion(images, t)

            graph_cond = gnn(graphs).to(device)

            t_normalized = t.float() / diffusion.timesteps
            predicted_noise = unet(noisy_images, t_normalized.to(device), graph_cond)

            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.4f}")


            if ( epoch + 1 ) % 5 == 0:
                torch.save({
                    "gnn": gnn.state_dict(), 
                    "unet": unet.state_dict(), 
                }, f"checkpoints/epoch_{epoch + 1}.pt")



if __name__ == "__main__":
    train()
