import torch
import torch.nn as nn
import torch.nn.functional as F

class ProportionalOddsHead(nn.Module):
    def __init__(self, hidden_dim: int, num_bins: int, dropout_p: float = 0.1, init_thresholds_range: float = 2.0):
        super().__init__()
        self.num_bins = num_bins
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, 1),
        )
        thresholds = torch.linspace(-init_thresholds_range, init_thresholds_range, num_bins - 1)
        self.thresholds = nn.Parameter(thresholds)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.score_layer(x)
        score = score.squeeze(-1)
        cumprobs = torch.sigmoid(self.thresholds - score.unsqueeze(-1))
        zeros = torch.zeros_like(cumprobs[..., :1])
        ones = torch.ones_like(cumprobs[..., :1])
        cumprobs_padded = torch.cat([zeros, cumprobs, ones], dim=-1)
        probs = torch.diff(cumprobs_padded, dim=-1)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, distance_weight: float = 1.0):
        super().__init__()
        self.distance_weight = distance_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, num_bins: int) -> torch.Tensor:
        predictions = predictions.reshape(-1, num_bins)
        targets = targets.reshape(-1).long()
        targets = torch.clamp(targets, 0, num_bins - 1)
        
        min_size = min(predictions.shape[0], targets.shape[0])
        predictions = predictions[:min_size]
        targets = targets[:min_size]
        
        loss = F.cross_entropy(predictions, targets, reduction='mean')
        
        if self.distance_weight > 0:
            probs = F.softmax(predictions, dim=-1)
            pred_classes = torch.argmax(probs, dim=-1)
            distance = torch.abs(pred_classes.float() - targets.float())
            weighted_loss = loss + self.distance_weight * distance.mean() / num_bins
            return weighted_loss
        
        return loss