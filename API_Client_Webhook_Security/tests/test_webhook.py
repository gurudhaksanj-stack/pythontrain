import hmac
import hashlib

from src.main import verify_signature


def test_valid_signature():
    payload = b"Hello Webhook"
    secret = "my_sample_webhook_secret"

    signature = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    assert verify_signature(payload, signature, secret) is True


def test_tampered_signature():
    payload = b"Hello Webhook"
    secret = "my_sample_webhook_secret"

    signature = "invalid_signature"

    assert verify_signature(payload, signature, secret) is False


def test_wrong_secret():
    payload = b"Hello Webhook"
    secret = "my_sample_webhook_secret"

    signature = hmac.new(
        b"wrong_secret",
        payload,
        hashlib.sha256
    ).hexdigest()

    assert verify_signature(payload, signature, secret) is False

    