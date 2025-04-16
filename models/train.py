# model/train.py
from torch.utils.data import DataLoader, WeightedRandomSampler
from telomere_dataset import TelomereDataset

def hybrid_train(config):
    dataset = TelomereDataset(
        csv_file='kaggle_input.csv',
        transform=DynamicNormalization()
    )
    
    sampler = WeightedRandomSampler(
        weights=dataset.class_weights, 
        num_samples=len(dataset),
        replacement=True
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        collate_fn=dynamic_padding
    )
    
    model = TemporalAttention()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'],
        weight_decay=0.01
    )
    
    for epoch in range(config['epochs']):
        for batch in loader:
            outputs = model(batch['features'])
            loss = focal_loss(outputs, batch['labels'])
            loss.backward()
            
            if (step+1) % config['accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()



class TemporalAttention(nn.Module):
    def __init__(self, input_dim=4, temp_heads=2):  # Reduced heads
        super().__init__()
        self.conv = nn.Conv1d(input_dim, 16)  # Reduced channels
        self.attention = nn.MultiheadAttention(16, temp_heads)
        self.fc = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1)
        )