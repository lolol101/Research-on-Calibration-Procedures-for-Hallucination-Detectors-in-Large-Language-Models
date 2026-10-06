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

    def register_input_scaling(self):
        """Add input standardization buffers, initialised to the identity.

        Calibration features lie in ``[0, 1]`` but vary across records on
        small and very different scales, so heads that consume them
        standardize every column first. The statistics are buffers: they
        travel with the state dict to every split the head is applied to.
        """
        self.register_buffer("input_mean", torch.zeros(self.in_features, device=self.device))
        self.register_buffer("input_std", torch.ones(self.in_features, device=self.device))

    def set_input_scaling(self, mean: torch.Tensor, std: torch.Tensor):
        """Standardize inputs as ``(features - mean) / std``.

        Args:
            mean: Per-feature mean, estimated on the training split.
            std: Per-feature standard deviation; must be positive.
        """
        with torch.no_grad():
            self.input_mean.copy_(mean.to(self.input_mean))
            self.input_std.copy_(std.to(self.input_std))

    def scale_inputs(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the standardization of ``set_input_scaling``."""
        return (features - self.input_mean) / self.input_std

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

    def __init__(self, in_features: int, device: torch.device, hidden_dim: int = 32, eps=1e-6):
        """
        Build a two-hidden-layer MLP with sigmoid output.

        Args:
            in_features: Input feature dimension.
            device: Parameter device.
            hidden_dim: Width of hidden layers.
            eps: Clamping bound for output probabilities; ``1 - eps`` must stay
                below 1 in float32.
        """
        super().__init__(in_features, device)
        
        self.eps = eps
        self.in_features = in_features
        self.device = device
        self.register_input_scaling()
        
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
        calibrated_confidence = self.net(self.scale_inputs(features)).squeeze(-1)
        calibrated_confidence = torch.clamp(calibrated_confidence, self.eps, 1 - self.eps)
        return calibrated_confidence

        
class MLPBetaCalibrationHead(CalibrationHead):
    """MLP sigmoid confidence followed by beta calibration."""

    def __init__(self, in_features: int, device: torch.device, hidden_dim: int = 32, eps=1e-6):
        """Initialize MLP trunk and learnable beta parameters ``a``, ``b``, ``c``.

        Args:
            in_features: Input feature dimension.
            device: Parameter device.
            hidden_dim: MLP hidden width.
            eps: Clamping bound before beta mapping; ``1 - eps`` must stay
                below 1 in float32, or ``log(1 - p)`` becomes ``-inf``.
        """
        super().__init__(in_features, device)
        
        self.eps = eps
        self.in_features = in_features
        self.device = device
        self.register_input_scaling()
        
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
        confidence = self.net(self.scale_inputs(features)).squeeze(-1)
        confidence = torch.clamp(confidence, self.eps, 1 - self.eps)

        a = torch.exp(self.log_a)
        b = torch.exp(self.log_b)
        calibrated_confidence = torch.sigmoid(
            self.c + a * torch.log(confidence) - b * torch.log(1 - confidence)
        )
        
        return calibrated_confidence

        
class TemperatureCalibrationHead(CalibrationHead):
    """Calibration via scaling logits by learable parameter in softmax procedure"""

    def __init__(self, in_features: int, device: torch.device, init_temperature: float = 1.0, eps: float = 1e-6):
        """
        Create a scalar temperature parameter.

        Args:
            in_features: Unused; kept for interface compatibility.
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
    """Sigmoid of a linear combination of features -> confidence, then beta calibration."""

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
        self.register_input_scaling()
        
        # Weighted sum
        self.weight_net = nn.Linear(in_features, 1, device=self.device)

        # Beta calibration
        self.log_a = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.log_b = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.c = nn.Parameter(torch.tensor(0.0, device=self.device))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Weighted feature sum, sigmoid, then beta calibration.

        The sigmoid keeps every record's gradient alive; clamping the raw sum
        to ``[eps, 1 - eps]`` would zero it wherever the sum falls outside.
        """
        confidence = torch.sigmoid(self.weight_net(self.scale_inputs(features)).squeeze(-1))
        confidence = torch.clamp(confidence, self.eps, 1 - self.eps)

        a = torch.exp(self.log_a)
        b = torch.exp(self.log_b)
        calibrated_confidence = torch.sigmoid(
            self.c + a * torch.log(confidence) - b * torch.log(1 - confidence)
        )
        
        return calibrated_confidence
        
