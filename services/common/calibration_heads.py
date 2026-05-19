from abc import ABC, abstractmethod
import torch
from torch import nn

class CalibrationHead(nn.Module, ABC):
    def __init__(self, in_features: int, device: torch.device):
        super().__init__()
        self.in_features = in_features
        self.device = device

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def calibrate(self, features: torch.Tensor, device=torch.device("cpu")) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(features.to(self.device)).to(device)

# MLP calibration: K per-head attention confidences + (not ATTN_ONLY) final-token confidence -> hidden layers -> Sigmoid
class MLPCalibrationHead(CalibrationHead):
    def __init__(self, in_features: int, device: torch.device, hidden_dim: int = 32, eps=1e-8):
        super().__init__(in_features, device)
        
        self.eps = eps
        self.in_features = in_features
        self.device = device
        
        # MLP
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        ).to(self.device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        calibrated_confidence = self.net(features).squeeze(-1)
        calibrated_confidence = torch.clamp(calibrated_confidence, self.eps, 1 - self.eps)
        return calibrated_confidence

    def calibrate(self, features: torch.Tensor, device=torch.device("cpu")) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(features.to(self.device)).to(device)
        
# MLP + Beta calibration: K per-head attention confidences + (not ATTN_ONLY) final-token confidence -> hidden layers -> Sigmoid -> Beta calibration
class MLPBetaCalibrationHead(CalibrationHead):
    def __init__(self, in_features: int, device: torch.device, hidden_dim: int = 32, eps=1e-8):
        super().__init__(in_features, device)
        
        self.eps = eps
        self.in_features = in_features
        self.device = device
        
        # MLP
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        ).to(self.device)
        
        # Beta calibration
        self.log_a = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.log_b = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.c = nn.Parameter(torch.tensor(0.0, device=self.device))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        confidence = self.net(features).squeeze(-1)
        confidence = torch.clamp(confidence, self.eps, 1 - self.eps)

        a = torch.exp(self.log_a)
        b = torch.exp(self.log_b)
        calibrated_confidence = torch.sigmoid(
            self.c + a * torch.log(confidence) - b * torch.log(1 - confidence)
        )
        
        return calibrated_confidence

    def calibrate(self, features: torch.Tensor, device=torch.device("cpu")) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(features.to(self.device)).to(device)
        
# Temperature calibration: [B, TOP_K] logits -> scaled logits
class TemperatureCalibrationHead(CalibrationHead):
    def __init__(self, in_features: int, device: torch.device, init_temperature: float = 1.0, eps: float = 1e-6):
        super().__init__(in_features, device)

        self.device = device
        self.eps = eps

        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(init_temperature, device=self.device, dtype=torch.float32))
        )

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.exp(self.log_temperature).clamp_min(self.eps)
        return logits / temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        scaled_logits = self.scale_logits(logits)
        return torch.softmax(scaled_logits, dim=-1)

    def calibrate(self, logits: torch.Tensor, device=torch.device("cpu")) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(logits.to(self.device)).to(device)

# Beta calibration: single probalistic feature -> Beta calibration
class BetaCalibrationHead(CalibrationHead):
    def __init__(self, in_features: int, device: torch.device, hidden_dim: int = 32, eps=1e-6):
        super().__init__(in_features, device)
        
        self.eps = eps
        self.in_features = in_features
        self.device = device
        
        # Beta calibration
        self.log_a = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.log_b = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.c = nn.Parameter(torch.tensor(0.0, device=self.device))

    def forward(self, confidence: torch.Tensor) -> torch.Tensor:
        confidence = torch.clamp(confidence, self.eps, 1 - self.eps)

        a = torch.exp(self.log_a)
        b = torch.exp(self.log_b)
        calibrated_confidence = torch.sigmoid(
            self.c + a * torch.log(confidence) - b * torch.log(1 - confidence)
        )
        
        return calibrated_confidence

    def calibrate(self, features: torch.Tensor, device=torch.device("cpu")) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(features.to(self.device)).to(device)

# Weighted Beta calibration:  K per-head attention confidences + (not ATTN_ONLY) final-token confidence -> weighted sum -> Beta calibration
class WeightedBetaCalibrationHead(CalibrationHead):
    def __init__(self, in_features: int, device: torch.device, hidden_dim: int = 32, eps=1e-6):
        super().__init__(in_features, device)
        
        self.eps = eps
        self.in_features = in_features
        self.device = device
        
        # Weighted sum
        self.weight_net = nn.Linear(in_features, 1, device=self.device)

        # Beta calibration
        self.log_a = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.log_b = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.c = nn.Parameter(torch.tensor(0.0, device=self.device))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        confidence = self.weight_net(features).squeeze(-1)
        confidence = torch.clamp(confidence, self.eps, 1 - self.eps)

        a = torch.exp(self.log_a)
        b = torch.exp(self.log_b)
        calibrated_confidence = torch.sigmoid(
            self.c + a * torch.log(confidence) - b * torch.log(1 - confidence)
        )
        
        return calibrated_confidence
    def calibrate(self, features: torch.Tensor, device=torch.device("cpu")) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(features.to(self.device)).to(device)
