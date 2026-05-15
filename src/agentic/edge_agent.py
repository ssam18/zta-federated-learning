"""
Edge Agent — paper-faithful implementation of the 5-module IIoT device
described in ZTA-FL Section IV.

Verbatim from the paper:

    Edge Agents: Each device consists of five functional modules:
      (1) A perception module for extracting feature information from flows,
          statistics and time-series;
      (2) A Local Intrusion Detection System (Local IDS) which utilizes an
          8-bit CNN-LSTM architecture (h_t = LSTM(CNN(x_t), h_{t-1}));
      (3) Adversarial Training via FGSM-PGD (x_adv = Clip(x + α·sign(∇_x L)))
          to generate adversarial samples at each device prior to sending
          them to the Fog Layer;
      (4) TPM-Based Attestation Module that generates tokens
          {ID_i, t, PCR, Sig_TPM} that are then sent to the Fog Layer;
      (5) Secure Communication between Edge Agents and the Fog Layer
          via Mutual TLS 1.3.

This module exposes those five modules as named attributes of the
:class:`EdgeAgent` so a deployment engineer can swap any of them
without touching the rest.  The module boundaries match the paper's
Figure 2 exactly and are referenced from the audit log so a reviewer
can replay every step.
"""

from __future__ import annotations

import copy
import logging
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn_lstm           import CNNLSTMClassifier
from src.security.attestation      import TPMDevice, AttestationToken
from src.security.adversarial      import adversarial_train_epoch
from src.agentic.config            import AgenticConfig


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module 1: Perception
# ---------------------------------------------------------------------------

class PerceptionModule:
    """
    Extracts features from flows / statistics / time-series.

    The public dataset CSVs are already pre-processed (PCA-reduced to 40
    features, min-max normalised) by ``src/utils/data_loader.py``.  The
    perception module here is the place where, in a real deployment, raw
    pcap → flow → feature extraction would live.  For the public release it
    is a thin pass-through with input-shape validation, but the interface is
    explicit so it can be swapped for a real-time tshark/Argus pipeline
    without touching downstream code.
    """

    def __init__(self, expected_n_features: int = 40) -> None:
        self.expected_n_features = expected_n_features

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() != 2 or X.shape[1] != self.expected_n_features:
            raise ValueError(
                f"perception module expects (N, {self.expected_n_features}) "
                f"input, got {tuple(X.shape)}"
            )
        return X.float()


# ---------------------------------------------------------------------------
# Module 4: TPM-based Attestation
# ---------------------------------------------------------------------------

class AttestationModule:
    """
    Wraps :class:`src.security.attestation.TPMDevice` per the paper spec.

    Tokens emitted here are the canonical :class:`AttestationToken` defined
    in ``src.security.attestation``, so the fog side can verify them with
    :meth:`AttestationAuthority.verify` directly without translation.
    """

    def __init__(self, agent_id: str, secret: str) -> None:
        self.agent_id = agent_id
        self._tpm     = TPMDevice(agent_id, secret)

    def generate_token(self, timestamp: Optional[float] = None) -> AttestationToken:
        ts = timestamp if timestamp is not None else time.time()
        return self._tpm.generate_token(timestamp=ts)

    @property
    def raw(self) -> TPMDevice:
        """Expose the underlying TPM device (for fog registration etc.)."""
        return self._tpm


# ---------------------------------------------------------------------------
# Module 5: Secure Communication (logical interface — no real TLS in process)
# ---------------------------------------------------------------------------

class SecureChannel:
    """
    Logical secure-channel interface.

    In production this wraps an mTLS 1.3 socket to the fog node.  In-process
    runs (this repo's experimental harness) treat the channel as a function
    call.  The interface is preserved so a deployment can drop in the real
    TLS wrapper without changing the rest of the agent code.
    """

    def __init__(self, fog_endpoint: str = "in-process") -> None:
        self.fog_endpoint = fog_endpoint

    def send(self, payload: Dict[str, Any]) -> None:
        # Real implementation: tls_socket.write(serialised payload)
        logger.debug(f"[secure_channel→{self.fog_endpoint}] sent payload "
                     f"agent={payload.get('agent_id')} round={payload.get('round')}")


# ---------------------------------------------------------------------------
# The Edge Agent itself: holds all 5 modules and the local FL loop
# ---------------------------------------------------------------------------

class EdgeAgent:
    """
    Paper-faithful edge agent.

    Public attributes match the paper's 5-module decomposition exactly so
    they can be referenced 1-1 from an audit trail or system diagram:

      * :attr:`perception`   — module (1)
      * :attr:`local_ids`    — module (2): the CNN-LSTM
      * :attr:`adv_training` — module (3): adversarial-augmented epoch fn
      * :attr:`attestation`  — module (4): TPM token generator
      * :attr:`secure_comm`  — module (5): logical mTLS channel

    Plus an autonomous *participation policy* (Section IV "Training
    Protocol"): the agent may self-quarantine when its local state is
    unhealthy and refuse to submit an update, mirroring the
    "agents must pass attestation before participating" semantics of the
    paper's TrustDB.
    """

    def __init__(
        self,
        agent_id:    str,
        n_features:  int,
        n_classes:   int,
        secret:      str,
        config:      Optional[AgenticConfig] = None,
        device:      str = "cpu",
        fog_endpoint: str = "in-process",
    ) -> None:
        self.agent_id   = agent_id
        self.device     = device
        self.config     = config or AgenticConfig()

        # Module 1: Perception
        self.perception   = PerceptionModule(expected_n_features=n_features)

        # Module 2: Local IDS — CNN-LSTM
        self.local_ids    = CNNLSTMClassifier(
            n_features=n_features, n_classes=n_classes,
        ).to(device)

        # Module 3: Adversarial training (function reference; preserved
        # explicitly so the deployment engineer can tune α, ε in one place)
        self.adv_training = adversarial_train_epoch

        # Module 4: TPM-based attestation
        self.attestation  = AttestationModule(agent_id=agent_id, secret=secret)

        # Module 5: Secure communication (logical mTLS interface)
        self.secure_comm  = SecureChannel(fog_endpoint=fog_endpoint)

        # Autonomous participation policy state
        self._loss_history: Deque[float] = deque(maxlen=5)
        self._self_quarantine_count = 0

    # ------------------------------------------------------------------
    # Pre-training: should I participate this round?
    # ------------------------------------------------------------------

    def decide_participation(
        self,
        local_data_size: int,
        attestation_ready: bool = True,
    ) -> Tuple[bool, str]:
        if not attestation_ready:
            self._self_quarantine_count += 1
            return False, "TPM module not ready"
        if local_data_size < 8:
            return False, f"insufficient local data ({local_data_size})"
        if self._loss_history:
            base = sum(self._loss_history) / len(self._loss_history)
            if base > 0 and self._loss_history[-1] > 1.5 * base:
                self._self_quarantine_count += 1
                return False, "self-quarantine: loss drift detected"
        return True, "nominal local state"

    # ------------------------------------------------------------------
    # Local training round
    # ------------------------------------------------------------------

    def local_round(
        self,
        global_state: Dict[str, torch.Tensor],
        Xi: torch.Tensor, yi: torch.Tensor,
        is_byzantine: bool = False,
        p_flip: float = 0.5,
    ) -> Tuple[Dict[str, torch.Tensor], AttestationToken, float]:
        """
        Run one full FL round at this edge:

          1. Perception module validates the input shape.
          2. Receive global state into Local IDS module.
          3. Adversarial-training module generates augmented batches.
          4. Local IDS trains for ``local_epochs`` epochs.
          5. Attestation module emits a fresh token.
          6. Secure-communication module ships {update, token} to fog.

        Returns ``(local_state_dict, attestation_token, training_loss)``.
        Byzantine clients in this harness flip a fraction of labels
        (paper Section VI attack scenario 1).
        """
        Xi = self.perception(Xi)

        # Receive global state
        self.local_ids.load_state_dict(global_state)
        self.local_ids.train()

        # Build loader (BatchNorm-safe: ≥2 per batch)
        n  = int(Xi.shape[0])
        bs = max(2, min(self.config.federation.batch_size, n // 2))
        drop = n >= bs * 3
        if is_byzantine:
            yi_eff = yi.clone()
            mask = torch.rand(len(yi)) < p_flip
            n_flip = int(mask.sum().item())
            if n_flip > 0:
                # Use n_classes from the model for proper bound
                n_classes = self.local_ids.n_classes
                yi_eff[mask] = torch.randint(0, n_classes, (n_flip,))
        else:
            yi_eff = yi
        loader = DataLoader(
            TensorDataset(Xi, yi_eff),
            batch_size=bs, shuffle=True, drop_last=drop,
        )

        # Adversarial-augmented local training (Section V.C)
        opt   = torch.optim.Adam(self.local_ids.parameters(),
                                 lr=self.config.federation.learning_rate)
        cfg   = self.config.adv_training
        loss  = 0.0
        for _ in range(self.config.federation.local_epochs):
            loss = self.adv_training(
                self.local_ids, loader, opt,
                adv_ratio=1.0 - cfg.clean_fraction,
                eps=cfg.pgd_eps, alpha=cfg.fgsm_alpha,
                n_iter=cfg.pgd_iters,
                device=self.device, use_pgd=True,
            )
        self._loss_history.append(float(loss))

        # Attestation + secure transmission
        token = self.attestation.generate_token()
        self.secure_comm.send({
            "agent_id": self.agent_id,
            "round":    None,           # filled in by caller
            "token":    token,
        })

        # Detach state dict for transmission
        local_state = {k: v.detach().clone()
                       for k, v in self.local_ids.state_dict().items()}
        return local_state, token, float(loss)

    # ------------------------------------------------------------------

    @property
    def self_quarantine_count(self) -> int:
        return self._self_quarantine_count
