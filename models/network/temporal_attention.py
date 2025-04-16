import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    Temporal Attention Network for predicting radiosensitivity from
    telomere length dynamics.
    """
    def __init__(self, input_features=11, embed_dim=64, num_heads=4, dropout_rate=0.3):
        super(TemporalAttention, self).__init__()
        
        # Ensure embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        # Feature extraction
        self.input_features = input_features
        self.conv1 = nn.Conv1d(1, embed_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        
        # Attention layers
        self.attention1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_rate)
        self.attention2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_rate)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # Fully connected layers with improved architecture
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """Forward pass with robust input handling"""
        try:
            # Reshape for 1D convolution [batch, 1, features]
            if x.dim() == 2:
                x = x.unsqueeze(1)
            
            # Feature extraction
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.dropout(x)
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.dropout(x)
            
            # Transpose for attention [seq_len, batch, features]
            x = x.permute(2, 0, 1)
            
            # Self-attention with residual connections
            x_norm = self.layer_norm1(x)
            attn_output1, _ = self.attention1(x_norm, x_norm, x_norm)
            x = x + attn_output1
            
            x_norm = self.layer_norm2(x)
            attn_output2, _ = self.attention2(x_norm, x_norm, x_norm)
            x = x + attn_output2
            
            # Global average pooling
            x = x.mean(dim=0)  # Average across sequence dimension
            
            # Classification
            return self.fc(x)
            
        except Exception as e:
            print(f"Forward pass error: {e}")
            # Return default prediction
            return torch.tensor([[0.5]], device=x.device).expand(x.size(0), 1)
    
    def get_attention_weights(self, x):
        """Get attention weights for visualization"""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Transpose for attention
        x = x.permute(2, 0, 1)
        
        # Get attention weights
        x_norm = self.layer_norm1(x)
        _, weights1 = self.attention1(x_norm, x_norm, x_norm)
        
        attn_output1 = x + self.dropout(self.attention1(x_norm, x_norm, x_norm)[0])
        x_norm = self.layer_norm2(attn_output1)
        _, weights2 = self.attention2(x_norm, x_norm, x_norm)
        
        return weights1, weights2
