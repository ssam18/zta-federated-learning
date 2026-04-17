"""
TPM-based zero-trust attestation for IIoT edge devices.

Each device in the federated learning topology must present a valid attestation
token before its model update is accepted by the fog aggregator.  Tokens are
cryptographically bound to the device's platform configuration registers (PCRs)
and include a freshness timestamp to prevent replay attacks.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Attestation token structure
# ---------------------------------------------------------------------------

@dataclass
class AttestationToken:
    """
    An attestation token issued by a TPM-equipped edge device.

    Attributes
    ----------
    device_id : str
        Unique identifier of the issuing device.
    pcr_digest : str
        SHA-256 digest of the device's platform configuration registers.
    timestamp : float
        Unix timestamp at which the token was generated.
    signature : str
        HMAC-SHA256 signature over (device_id + pcr_digest + timestamp).
    nonce : str
        Random nonce included to prevent replay attacks.
    """

    device_id: str
    pcr_digest: str
    timestamp: float
    signature: str
    nonce: str


# ---------------------------------------------------------------------------
# TPM device emulation
# ---------------------------------------------------------------------------

class TPMDevice:
    """
    Represents a TPM 2.0-equipped IIoT edge device that can issue attestation
    tokens to the fog aggregator.

    In a physical deployment this class wraps the TPM2-TSS software stack.
    The signing key is derived from the device's Attestation Identity Key (AIK)
    provisioned during manufacturing.

    Parameters
    ----------
    device_id : str
        Unique identifier for this device (e.g., MAC address or serial number).
    aik_secret : str
        Shared Attestation Identity Key secret (pre-provisioned out-of-band).
    pcr_values : dict of str → str, optional
        Current PCR register contents.  If not supplied, default known-good
        values are used.
    """

    # Known-good PCR snapshot for a clean Raspberry Pi 4 + IIoT firmware build
    _DEFAULT_PCR: Dict[str, str] = {
        "PCR0": "0000000000000000000000000000000000000000000000000000000000000000",
        "PCR1": "b2a73e0d5c64ef9d8ea3f8b9f7c6a1d2e5f3b9c4a7d6e2f1b8c3a4d5e6f7b9c2",
        "PCR7": "3f5a6b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a",
    }

    def __init__(
        self,
        device_id: str,
        aik_secret: str,
        pcr_values: Optional[Dict[str, str]] = None,
    ) -> None:
        self.device_id = device_id
        self._aik_secret = aik_secret.encode()
        self._pcr_values = pcr_values if pcr_values is not None else dict(self._DEFAULT_PCR)

    # ------------------------------------------------------------------
    def _compute_pcr_digest(self) -> str:
        """Compute a SHA-256 digest over the concatenated PCR values."""
        pcr_concat = "".join(
            f"{k}:{v}" for k, v in sorted(self._pcr_values.items())
        )
        return hashlib.sha256(pcr_concat.encode()).hexdigest()

    # ------------------------------------------------------------------
    def _sign(self, message: str) -> str:
        """Compute an HMAC-SHA256 signature over the given message."""
        return hmac.new(self._aik_secret, message.encode(), hashlib.sha256).hexdigest()

    # ------------------------------------------------------------------
    def generate_token(self, timestamp: Optional[float] = None) -> AttestationToken:
        """
        Issue a fresh attestation token for presentation to the fog aggregator.

        The token encodes the current PCR digest, a fresh timestamp, and a
        device-signed HMAC to allow the aggregator to verify integrity and
        freshness.

        Parameters
        ----------
        timestamp : float, optional
            Unix timestamp to embed in the token.  Defaults to the current
            system time (``time.time()``).

        Returns
        -------
        AttestationToken
            A signed, timestamped attestation token.
        """
        if timestamp is None:
            timestamp = time.time()

        pcr_digest = self._compute_pcr_digest()

        # Nonce: first 16 hex chars of SHA-256(device_id + timestamp)
        nonce_src = f"{self.device_id}{timestamp}"
        nonce = hashlib.sha256(nonce_src.encode()).hexdigest()[:32]

        # Message to sign: device_id | pcr_digest | timestamp | nonce
        message = f"{self.device_id}|{pcr_digest}|{timestamp:.6f}|{nonce}"
        signature = self._sign(message)

        return AttestationToken(
            device_id=self.device_id,
            pcr_digest=pcr_digest,
            timestamp=timestamp,
            signature=signature,
            nonce=nonce,
        )


# ---------------------------------------------------------------------------
# Attestation authority (fog aggregator side)
# ---------------------------------------------------------------------------

class AttestationAuthority:
    """
    Fog-side Attestation Authority that verifies device tokens and maintains
    per-device trust scores.

    The authority enforces:
    - Signature validity (HMAC check with the pre-shared AIK secret).
    - Token freshness (timestamp within ``max_age_seconds`` of current time).
    - Replay prevention (each nonce may only be accepted once per session).
    - Expected PCR digest comparison against a stored golden value.

    Parameters
    ----------
    aik_registry : dict of str → str
        Mapping from device ID to its pre-shared AIK secret.
    pcr_registry : dict of str → str, optional
        Mapping from device ID to its expected PCR digest.  If a device is
        not in this registry its PCR check is skipped.
    max_age_seconds : float
        Maximum acceptable token age in seconds.
    trust_decay : float
        Factor by which the trust score decays on each failed verification.
    """

    def __init__(
        self,
        aik_registry: Dict[str, str],
        pcr_registry: Optional[Dict[str, str]] = None,
        max_age_seconds: float = 30.0,
        trust_decay: float = 0.5,
    ) -> None:
        self._aik_registry = {k: v.encode() for k, v in aik_registry.items()}
        self._pcr_registry = pcr_registry or {}
        self.max_age_seconds = max_age_seconds
        self.trust_decay = trust_decay

        # Trust scores start at 1.0 for all registered devices
        self._trust_scores: Dict[str, float] = {
            dev: 1.0 for dev in aik_registry
        }
        self._seen_nonces: set[str] = set()

    # ------------------------------------------------------------------
    def verify(
        self,
        token: AttestationToken,
        current_time: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Verify an attestation token presented by an edge device.

        Parameters
        ----------
        token : AttestationToken
            The token to verify.
        current_time : float, optional
            Reference time for freshness check.  Defaults to ``time.time()``.

        Returns
        -------
        (bool, str)
            ``(True, "OK")`` if all checks pass, or ``(False, reason)`` if any
            check fails.
        """
        if current_time is None:
            current_time = time.time()

        device_id = token.device_id

        # 1. Device must be registered
        if device_id not in self._aik_registry:
            return False, f"Unknown device: {device_id}"

        # 2. Freshness check
        age = current_time - token.timestamp
        if age < 0 or age > self.max_age_seconds:
            self.update_trust(device_id, delta=-0.1)
            return False, f"Token age {age:.1f}s outside acceptable window."

        # 3. Replay check
        if token.nonce in self._seen_nonces:
            self.update_trust(device_id, delta=-0.2)
            return False, "Replay detected: nonce already seen."

        # 4. Signature verification
        message = f"{device_id}|{token.pcr_digest}|{token.timestamp:.6f}|{token.nonce}"
        expected_sig = hmac.new(
            self._aik_registry[device_id], message.encode(), hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(token.signature, expected_sig):
            self.update_trust(device_id, delta=-0.3)
            return False, "Signature mismatch: token integrity check failed."

        # 5. PCR digest check (if golden value is registered)
        if device_id in self._pcr_registry:
            if token.pcr_digest != self._pcr_registry[device_id]:
                self.update_trust(device_id, delta=-0.4)
                return False, "PCR mismatch: platform state has changed."

        # All checks passed
        self._seen_nonces.add(token.nonce)
        self.update_trust(device_id, delta=0.05)
        return True, "OK"

    # ------------------------------------------------------------------
    def update_trust(self, device_id: str, delta: float) -> None:
        """
        Adjust the trust score for a device by ``delta``.

        Trust scores are bounded to [0.0, 1.0].  A score below 0.3 causes the
        device to be flagged as untrusted and excluded from aggregation.

        Parameters
        ----------
        device_id : str
            The device whose score to update.
        delta : float
            Additive change to the trust score (positive = reward, negative = penalty).
        """
        if device_id not in self._trust_scores:
            self._trust_scores[device_id] = 1.0

        new_score = self._trust_scores[device_id] + delta
        self._trust_scores[device_id] = max(0.0, min(1.0, new_score))

    # ------------------------------------------------------------------
    def get_trust_score(self, device_id: str) -> float:
        """Return the current trust score for ``device_id``, or 0.0 if unknown."""
        return self._trust_scores.get(device_id, 0.0)

    # ------------------------------------------------------------------
    def is_trusted(self, device_id: str, threshold: float = 0.3) -> bool:
        """Return True if the device's trust score meets the threshold."""
        return self.get_trust_score(device_id) >= threshold

    # ------------------------------------------------------------------
    def trusted_devices(self, threshold: float = 0.3) -> list[str]:
        """Return a list of device IDs whose trust score meets the threshold."""
        return [d for d, s in self._trust_scores.items() if s >= threshold]
