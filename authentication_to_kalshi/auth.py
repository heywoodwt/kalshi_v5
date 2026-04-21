"""
Authentication for Kalshi API using RSA-PSS signatures.

Kalshi requires RSA-PSS signed requests with:
- API key in header
- Timestamp in header (milliseconds since epoch)
- Signature of: timestamp + method + path

The signature uses SHA-256 hash with PSS padding for security.
"""
import base64
import time
from typing import Dict

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

# Kalshi WebSocket API path
WS_API_PATH = "/trade-api/ws/v2"


def load_private_key(path: str) -> RSAPrivateKey:
    """
    Load RSA private key from PEM file.

    The key is used to sign API requests. Must match the public key
    registered with your Kalshi API account.

    Args:
        path: File path to PEM-encoded private key

    Returns:
        RSA private key object
    """
    with open(path, "rb") as f:
        # Load PEM key with no password encryption
        return serialization.load_pem_private_key(f.read(), password=None)


def sign_pss(key: RSAPrivateKey, message: str) -> str:
    """
    Sign a message using RSA-PSS with SHA-256.

    PSS (Probabilistic Signature Scheme) is more secure than PKCS#1 v1.5
    because it uses randomization. Kalshi requires this signature scheme.

    Args:
        key: RSA private key
        message: String to sign (will be UTF-8 encoded)

    Returns:
        Base64-encoded signature string
    """
    # Sign with PSS padding and SHA-256 hash
    signature = key.sign(
        message.encode(),  # Convert string to bytes
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),  # Mask generation function
            salt_length=padding.PSS.MAX_LENGTH,  # Maximum salt for security
        ),
        hashes.SHA256(),  # Hash algorithm
    )

    # Encode signature as base64 string for HTTP header
    return base64.b64encode(signature).decode()


def make_ws_headers(api_key: str, key: RSAPrivateKey) -> Dict[str, str]:
    """
    Generate authentication headers for Kalshi WebSocket connection.

    Kalshi authenticates WebSocket connections using signed headers.
    The signature proves we own the private key matching our API key.

    Signature message format: timestamp + method + path
    Example: "1713547200000GET/trade-api/ws/v2"

    Args:
        api_key: Your Kalshi API key
        key: Your RSA private key

    Returns:
        Dict of HTTP headers for WebSocket connection
    """
    # Current time in milliseconds (Kalshi requires this precision)
    timestamp_ms = str(int(time.time() * 1000))

    # Construct message to sign: timestamp + HTTP method + API path
    message = timestamp_ms + "GET" + WS_API_PATH

    # Generate signature
    signature = sign_pss(key, message)

    # Return headers required by Kalshi
    return {
        "KALSHI-ACCESS-KEY": api_key,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
    }
