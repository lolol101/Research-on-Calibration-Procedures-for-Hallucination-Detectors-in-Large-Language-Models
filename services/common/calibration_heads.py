from abc import ABC, abstractmethod
import torch
from torch import nn

class CalibrationHead(nn.Module, ABC):
    """Abstract calibration module mapping features to calibrated confidences."""

    def __init__(self, in_features: int, device: torch.device):
        """Store input size and target device.

        Args:
            in_features: Number of input features.
            device: torch.device to perform computations on.
        """
        super().__init__()
        self.in_features = in_features
        self.device = device

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map raw features to calibrated outputs."""
        raise NotImplementedError

    def calibrate(self, features: torch.Tensor, device=torch.device("cpu")) -> torch.Tensor:
        """Run ``forward`` in eval mode without gradients.

        Args:
            features: Input tensor on any device.
            device: Device for the returned tensor.

        Returns:
            Calibrated predictions on ``device``.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(features.to(self.device)).to(device)

class MLPCalibrationHead(CalibrationHead):
    """MLP: attention (+ optional final) features -> sigmoid calibrated probability."""

    def __init__(self, in_features: int, device: torch.device, hidden_dim: int = 32, eps=1e-8):
        """
        Build a two-hidden-layer MLP with sigmoid output.

        Args:
            in_features: Input feature dimension.
            device: Parameter device.
            hidden_dim: Width of hidden layers.
            eps: Clamping bound for output probabilities.
        """
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
        """Return clamped MLP confidence for each row of ``features``."""
        calibrated_confidence = self.net(features).squeeze(-1)
        calibrated_confidence = torch.clamp(calibrated_confidence, self.eps, 1 - self.eps)
        return calibrated_confidence

        
class MLPBetaCalibrationHead(CalibrationHead):
    """MLP sigmoid confidence followed by beta calibration."""

    def __init__(self, in_features: int, device: torch.device, hidden_dim: int = 32, eps=1e-8):
        """Initialize MLP trunk and learnable beta parameters ``a``, ``b``, ``c``.

        Args:
            in_features: Input feature dimension.
            device: Parameter device.
            hidden_dim: MLP hidden width.
            eps: Clamping bound before beta mapping.
        """
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
        """MLP confidence then beta-calibrated probability."""
        confidence = self.net(features).squeeze(-1)
        confidence = torch.clamp(confidence, self.eps, 1 - self.eps)

        a = torch.exp(self.log_a)
        b = torch.exp(self.log_b)
        calibrated_confidence = torch.sigmoid(
            self.c + a * torch.log(confidence) - b * torch.log(1 - confidence)
        )
        
        return calibrated_confidence

        
class TemperatureCalibrationHead(CalibrationHead):
    """Calibration via scaling logits by learable parameter in softmax procedure"""

    def __init__(self, device: torch.device, init_temperature: float = 1.0, eps: float = 1e-6):
        """
        Create a scalar temperature parameter.

        Args:
            device: torch.device to perform computations on.
            init_temperature: Initial temperature value (> 0).
            eps: Clamping parameter.
        """
        super().__init__(1, device)

        self.device = device
        self.eps = eps

        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(init_temperature, device=self.device, dtype=torch.float32))
        )

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Divide logits by the learned positive temperature."""
        temperature = torch.exp(self.log_temperature).clamp_min(self.eps)
        return logits / temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Softmax over temperature-scaled logits."""
        scaled_logits = self.scale_logits(logits)
        return torch.softmax(scaled_logits, dim=-1)


class BetaCalibrationHead(CalibrationHead):
    """Beta calibration mapping raw confidences to calibrated probabilities."""

    def __init__(self, in_features: int, device: torch.device, eps=1e-6):
        """Learn beta parameters ``a``, ``b``, ``c`` (``hidden_dim`` unused).

        Args:
            in_features: Unused; kept for interface compatibility.
            device: Parameter device.
            eps: Clamping bound on input confidence.
        """
        super().__init__(in_features, device)
        
        self.eps = eps
        self.in_features = in_features
        self.device = device
        
        # Beta calibration
        self.log_a = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.log_b = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.c = nn.Parameter(torch.tensor(0.0, device=self.device))

    def forward(self, confidence: torch.Tensor) -> torch.Tensor:
        """Apply beta calibration to each confidence value."""
        confidence = torch.clamp(confidence, self.eps, 1 - self.eps)

        a = torch.exp(self.log_a)
        b = torch.exp(self.log_b)
        calibrated_confidence = torch.sigmoid(
            self.c + a * torch.log(confidence) - b * torch.log(1 - confidence)
        )
        
        return calibrated_confidence


class WeightedBetaCalibrationHead(CalibrationHead):
    """Linear combination of features -> confidence, then beta calibration."""

    def __init__(self, in_features: int, device: torch.device, eps=1e-6):
        """
        Initialize feature weights and beta parameters.

        Args:
            in_features: Input feature dimension for ``weight_net``.
            device: Parameter device.
            eps: Clamping bound on the mixed confidence.
        """
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
        """Weighted feature sum, clamp, then beta calibration."""
        confidence = self.weight_net(features).squeeze(-1)
        confidence = torch.clamp(confidence, self.eps, 1 - self.eps)

        a = torch.exp(self.log_a)
        b = torch.exp(self.log_b)
        calibrated_confidence = torch.sigmoid(
            self.c + a * torch.log(confidence) - b * torch.log(1 - confidence)
        )
        
        return calibrated_confidence
        
