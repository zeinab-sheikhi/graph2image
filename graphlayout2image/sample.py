import torch
from PIL import Image
import matplotlib.pyplot as plt

@torch.no_grad()
def sample(gnn, unet, diffusion, scene_graph, device='cuda'):
    """Generate image from scene graph"""
    gnn.eval()
    unet.eval()
    
    # Encode graph
    graph_cond = gnn([scene_graph]).to(device)
    
    # Start from pure noise
    x = torch.randn(1, 3, 32, 32).to(device)
    
    # Reverse diffusion
    for t in reversed(range(0, diffusion.timesteps, 50)):  # Sample every 50 steps
        t_batch = torch.tensor([t]).to(device)
        t_normalized = t_batch.float() / diffusion.timesteps
        
        # Predict noise
        predicted_noise = unet(x, t_normalized, graph_cond)
        
        # Denoise step
        alpha_t = diffusion.alphas[t]
        alpha_cumprod_t = diffusion.alphas_cumprod[t]
        beta_t = diffusion.betas[t]
        
        x = (1 / alpha_t.sqrt()) * (
            x - (beta_t / (1 - alpha_cumprod_t).sqrt()) * predicted_noise
        )
        
        # Add noise if not final step
        if t > 0:
            noise = torch.randn_like(x)
            x = x + (beta_t.sqrt() * noise)
    
    # Convert to image
    x = x.clamp(0, 1)
    img = x[0].permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype('uint8')

# Test sampling
test_graphs = [
    {
        'nodes': [
            {'shape': 'circle', 'color': 'red'},
            {'shape': 'square', 'color': 'blue'}
        ],
        'edges': [{'relation': 'left_of'}]
    },
    {
        'nodes': [
            {'shape': 'triangle', 'color': 'green'},
            {'shape': 'circle', 'color': 'yellow'}
        ],
        'edges': [{'relation': 'above'}]
    }
]

# Load checkpoint
checkpoint = torch.load('checkpoint_epoch20.pt')
gnn.load_state_dict(checkpoint['gnn'])
unet.load_state_dict(checkpoint['unet'])

# Generate samples
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, graph in enumerate(test_graphs):
    for j in range(3):  # 3 samples per graph
        img = sample(gnn, unet, diffusion, graph)
        axes[i, j].imshow(img)
        axes[i, j].axis('off')
        if j == 0:
            # Show graph description
            relation = graph['edges'][0]['relation']
            axes[i, j].set_title(f"{graph['nodes'][0]['color']} {graph['nodes'][0]['shape']}\n{relation}\n{graph['nodes'][1]['color']} {graph['nodes'][1]['shape']}", 
                                fontsize=8)

plt.tight_layout()
plt.savefig('results.png', dpi=150)
