import os
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import hmac
import hashlib

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")


# Create requests session with retry logic
def create_session():
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session = requests.Session()
    session.mount("https://", adapter)

    return session


# Fetch users from API
def get_users():
    session = create_session()

    response = session.get(
        "https://jsonplaceholder.typicode.com/users"
    )

    response.raise_for_status()

    return response.json()


# Print first 5 users
def display_users(users):
    for user in users[:5]:
        print(f"{user.get('id')} - {user.get('name')} - {user.get('email')}")


# Print first user's nested details
def display_first_user_details(users):
    user = users[0]

    address = user.get("address", {})
    geo = address.get("geo", {})
    company = user.get("company", {})

    city = address.get("city", "Not Available")
    latitude = geo.get("lat", "Not Available")
    zip_code = address.get("zipcode", "Not Available")
    company_name = company.get("name", "Not Available")

    print(
        f"City: {city} | Latitude: {latitude} | "
        f"Zip: {zip_code} | Company: {company_name}"
    )


# Verify webhook signature
def verify_signature(payload, signature, secret):
    expected_signature = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected_signature, signature)


# Main function
def main():
    users = get_users()

    display_users(users)

    print()

    display_first_user_details(users)

    print()

    payload = b"Hello Webhook"

    valid_signature = hmac.new(
        WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    tampered_signature = "invalid_signature"

    wrong_secret_signature = hmac.new(
        b"wrong_secret",
        payload,
        hashlib.sha256
    ).hexdigest()

    print(
        f"Valid: {verify_signature(payload, valid_signature, WEBHOOK_SECRET)}"
        f" | Tampered: {verify_signature(payload, tampered_signature, WEBHOOK_SECRET)}"
        f" | Wrong secret: {verify_signature(payload, wrong_secret_signature, WEBHOOK_SECRET)}"
    )


if __name__ == "__main__":
    main()


    