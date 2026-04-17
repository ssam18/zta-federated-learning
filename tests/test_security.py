"""
Unit tests for the TPM-based zero-trust attestation module.

Tests cover token generation, signature verification, freshness enforcement,
replay attack detection, and impersonation prevention.
"""

import time
import unittest

from src.security.attestation import (
    AttestationAuthority,
    AttestationToken,
    TPMDevice,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_device(device_id: str = "dev-001", secret: str = "secret-aik") -> TPMDevice:
    return TPMDevice(device_id=device_id, aik_secret=secret)


def _make_authority(
    device_id: str = "dev-001",
    secret: str = "secret-aik",
    max_age: float = 30.0,
) -> AttestationAuthority:
    return AttestationAuthority(
        aik_registry={device_id: secret},
        max_age_seconds=max_age,
    )


# ---------------------------------------------------------------------------
# TPMDevice tests
# ---------------------------------------------------------------------------

class TestTPMDevice(unittest.TestCase):

    def test_token_fields_populated(self):
        """Generated token should have all required fields set."""
        dev = _make_device()
        tok = dev.generate_token()
        self.assertEqual(tok.device_id, "dev-001")
        self.assertIsInstance(tok.pcr_digest, str)
        self.assertGreater(len(tok.pcr_digest), 0)
        self.assertIsInstance(tok.timestamp, float)
        self.assertIsInstance(tok.signature, str)
        self.assertGreater(len(tok.signature), 0)
        self.assertIsInstance(tok.nonce, str)
        self.assertGreater(len(tok.nonce), 0)

    def test_custom_timestamp_embedded(self):
        """Custom timestamp should appear in the generated token."""
        dev = _make_device()
        ts = 1_700_000_000.0
        tok = dev.generate_token(timestamp=ts)
        self.assertAlmostEqual(tok.timestamp, ts, places=3)

    def test_pcr_digest_is_deterministic(self):
        """Two tokens from the same device should have the same PCR digest."""
        dev = _make_device()
        tok1 = dev.generate_token(timestamp=1000.0)
        tok2 = dev.generate_token(timestamp=2000.0)
        self.assertEqual(tok1.pcr_digest, tok2.pcr_digest)

    def test_different_timestamps_different_nonces(self):
        """Tokens generated at different times should have different nonces."""
        dev = _make_device()
        tok1 = dev.generate_token(timestamp=1000.0)
        tok2 = dev.generate_token(timestamp=2000.0)
        self.assertNotEqual(tok1.nonce, tok2.nonce)

    def test_different_secrets_different_signatures(self):
        """Devices with different AIK secrets should produce different signatures."""
        dev_a = TPMDevice("dev-A", aik_secret="secret-A")
        dev_b = TPMDevice("dev-B", aik_secret="secret-B")
        ts = 9999.0
        tok_a = dev_a.generate_token(timestamp=ts)
        tok_b = dev_b.generate_token(timestamp=ts)
        self.assertNotEqual(tok_a.signature, tok_b.signature)


# ---------------------------------------------------------------------------
# AttestationAuthority — successful verification
# ---------------------------------------------------------------------------

class TestAttestationAuthorityVerify(unittest.TestCase):

    def test_valid_token_accepted(self):
        """A freshly issued, correctly signed token should pass verification."""
        dev = _make_device()
        auth = _make_authority()
        ts = time.time()
        tok = dev.generate_token(timestamp=ts)
        ok, reason = auth.verify(tok, current_time=ts + 1.0)
        self.assertTrue(ok, f"Valid token rejected: {reason}")
        self.assertEqual(reason, "OK")

    def test_trust_score_nondecreasing_on_success(self):
        """Trust score should not decrease after a successful verification."""
        dev = _make_device()
        auth = _make_authority()
        # Lower the score below 1.0 first so there is room to increase
        auth.update_trust("dev-001", delta=-0.3)
        score_before = auth.get_trust_score("dev-001")
        ts = time.time()
        tok = dev.generate_token(timestamp=ts)
        auth.verify(tok, current_time=ts + 0.1)
        self.assertGreaterEqual(auth.get_trust_score("dev-001"), score_before)


# ---------------------------------------------------------------------------
# AttestationAuthority — rejection cases
# ---------------------------------------------------------------------------

class TestAttestationAuthorityRejection(unittest.TestCase):

    def test_unknown_device_rejected(self):
        """Token from an unregistered device should be rejected."""
        unknown_dev = TPMDevice("unknown-dev", aik_secret="any-secret")
        auth = _make_authority()
        ts = time.time()
        tok = unknown_dev.generate_token(timestamp=ts)
        ok, reason = auth.verify(tok, current_time=ts)
        self.assertFalse(ok)
        self.assertIn("Unknown", reason)

    def test_stale_token_rejected(self):
        """Token older than max_age_seconds should be rejected."""
        dev = _make_device()
        auth = _make_authority(max_age=10.0)
        ts = time.time() - 60.0  # 60 seconds old
        tok = dev.generate_token(timestamp=ts)
        ok, reason = auth.verify(tok, current_time=time.time())
        self.assertFalse(ok)

    def test_future_token_rejected(self):
        """Token with a future timestamp should be rejected."""
        dev = _make_device()
        auth = _make_authority(max_age=5.0)
        ts = time.time() + 3600.0  # 1 hour in the future
        tok = dev.generate_token(timestamp=ts)
        ok, reason = auth.verify(tok, current_time=time.time())
        self.assertFalse(ok)

    def test_replay_attack_rejected(self):
        """Presenting the same token twice should trigger replay detection."""
        dev = _make_device()
        auth = _make_authority()
        ts = time.time()
        tok = dev.generate_token(timestamp=ts)

        ok1, _ = auth.verify(tok, current_time=ts + 1.0)
        self.assertTrue(ok1, "First presentation should succeed.")

        ok2, reason = auth.verify(tok, current_time=ts + 2.0)
        self.assertFalse(ok2, "Second presentation of same token should fail.")
        self.assertIn("Replay", reason)

    def test_tampered_signature_rejected(self):
        """Mutating the signature should cause verification to fail."""
        dev = _make_device()
        auth = _make_authority()
        ts = time.time()
        tok = dev.generate_token(timestamp=ts)
        # Tamper with the signature
        tampered = AttestationToken(
            device_id=tok.device_id,
            pcr_digest=tok.pcr_digest,
            timestamp=tok.timestamp,
            signature="deadbeef" * 8,
            nonce=tok.nonce,
        )
        ok, reason = auth.verify(tampered, current_time=ts + 1.0)
        self.assertFalse(ok)
        self.assertIn("Signature", reason)

    def test_pcr_mismatch_rejected(self):
        """A token with a different PCR digest should be rejected when golden value is registered."""
        dev = _make_device()
        pcr_registry = {"dev-001": "golden_digest_value_that_wont_match"}
        auth = AttestationAuthority(
            aik_registry={"dev-001": "secret-aik"},
            pcr_registry=pcr_registry,
            max_age_seconds=30.0,
        )
        ts = time.time()
        tok = dev.generate_token(timestamp=ts)
        ok, reason = auth.verify(tok, current_time=ts + 1.0)
        self.assertFalse(ok)
        self.assertIn("PCR", reason)

    def test_impersonation_rejected(self):
        """Device B cannot forge a valid token for device A with a different key."""
        dev_a = TPMDevice("dev-001", aik_secret="correct-secret")
        impersonator = TPMDevice("dev-001", aik_secret="wrong-secret")
        auth = _make_authority(secret="correct-secret")
        ts = time.time()
        forged_tok = impersonator.generate_token(timestamp=ts)
        ok, reason = auth.verify(forged_tok, current_time=ts + 1.0)
        self.assertFalse(ok, "Impersonation attempt should be rejected.")
        self.assertIn("Signature", reason)


# ---------------------------------------------------------------------------
# Trust score management
# ---------------------------------------------------------------------------

class TestTrustScoreManagement(unittest.TestCase):

    def test_initial_trust_is_one(self):
        """All registered devices should start with trust score 1.0."""
        auth = AttestationAuthority(
            aik_registry={"dev-A": "s1", "dev-B": "s2"}
        )
        self.assertAlmostEqual(auth.get_trust_score("dev-A"), 1.0)
        self.assertAlmostEqual(auth.get_trust_score("dev-B"), 1.0)

    def test_unknown_device_score_is_zero(self):
        """Unknown devices should return trust score 0.0."""
        auth = _make_authority()
        self.assertEqual(auth.get_trust_score("ghost-device"), 0.0)

    def test_trust_clamped_at_one(self):
        """Trust score should not exceed 1.0."""
        auth = _make_authority()
        for _ in range(100):
            auth.update_trust("dev-001", delta=0.5)
        self.assertLessEqual(auth.get_trust_score("dev-001"), 1.0)

    def test_trust_clamped_at_zero(self):
        """Trust score should not drop below 0.0."""
        auth = _make_authority()
        for _ in range(100):
            auth.update_trust("dev-001", delta=-0.5)
        self.assertGreaterEqual(auth.get_trust_score("dev-001"), 0.0)

    def test_trusted_devices_filter(self):
        """trusted_devices() should respect the threshold."""
        auth = AttestationAuthority(
            aik_registry={"dev-A": "s1", "dev-B": "s2", "dev-C": "s3"}
        )
        auth.update_trust("dev-B", delta=-1.0)  # Score = 0.0
        trusted = auth.trusted_devices(threshold=0.3)
        self.assertIn("dev-A", trusted)
        self.assertIn("dev-C", trusted)
        self.assertNotIn("dev-B", trusted)

    def test_is_trusted(self):
        """is_trusted should correctly evaluate against threshold."""
        auth = _make_authority()
        auth.update_trust("dev-001", delta=-0.8)  # Score = 0.2
        self.assertFalse(auth.is_trusted("dev-001", threshold=0.3))
        auth.update_trust("dev-001", delta=0.2)   # Score = 0.4
        self.assertTrue(auth.is_trusted("dev-001", threshold=0.3))


if __name__ == "__main__":
    unittest.main()
