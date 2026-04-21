# #!/usr/bin/env python3
# """
# setup_webhook.py
# ----------------
# Run once per repo to register the GitHub/Bitbucket webhook
# pointing at your central Hetzner indexer.

# Usage:
#     python setup_webhook.py --repo https://github.com/org/myrepo
#     python setup_webhook.py --repo https://bitbucket.org/org/myrepo

# Requires:
#     GITHUB_TOKEN  or  BITBUCKET_USER + BITBUCKET_APP_PASSWORD  in .env
#     WEBHOOK_URL   = https://your-hetzner-ip/webhook
#     WEBHOOK_SECRET = any random string (same one in your server .env)
# """

# import os, sys, json, argparse, requests, secrets
# from dotenv import load_dotenv

# load_dotenv()

# WEBHOOK_URL    = os.getenv("WEBHOOK_URL")          # e.g. http://1.2.3.4:8000/webhook
# WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")       # shared secret for signature verification
# GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN")         # personal access token, repo scope
# BB_USER        = os.getenv("BITBUCKET_USER")
# BB_PASS        = os.getenv("BITBUCKET_API_TOKEN")


# def register_github(owner: str, repo: str):
#     url = f"https://api.github.com/repos/{owner}/{repo}/hooks"
#     payload = {
#         "name": "web",
#         "active": True,
#         "events": ["push", "pull_request"],
#         "config": {
#             "url":          WEBHOOK_URL,
#             "content_type": "json",
#             "secret":       WEBHOOK_SECRET,
#             "insecure_ssl": "0",
#         },
#     }
#     r = requests.post(url, json=payload,
#                       headers={"Authorization": f"token {GITHUB_TOKEN}",
#                                "Accept": "application/vnd.github+json"})
#     if r.status_code == 201:
#         print(f"✅ GitHub webhook registered → {WEBHOOK_URL}")
#         print(f"   Hook ID: {r.json()['id']}  (save this if you want to delete it later)")
#     elif r.status_code == 422:
#         print("⚠️  Webhook already exists for this URL — skipping.")
#     else:
#         print(f"❌ GitHub error {r.status_code}: {r.text}")
#         sys.exit(1)


# def register_bitbucket(workspace: str, repo: str):
#     url = f"https://api.bitbucket.org/2.0/repositories/{workspace}/{repo}/hooks"
#     payload = {
#         "description": "Code RAG indexer",
#         "url":         WEBHOOK_URL,
#         "active":      True,
#         "secret":      WEBHOOK_SECRET,
#         "events":      ["repo:push", "pullrequest:created", "pullrequest:updated"],
#     }
#     r = requests.post(url, json=payload, auth=(BB_USER, BB_PASS))
#     if r.status_code == 201:
#         print(f"✅ Bitbucket webhook registered → {WEBHOOK_URL}")
#     elif r.status_code == 409:
#         print("⚠️  Webhook already exists — skipping.")
#     else:
#         print(f"❌ Bitbucket error {r.status_code}: {r.text}")
#         sys.exit(1)


# def parse_repo_url(url: str):
#     # returns ("github"|"bitbucket", "owner", "repo")
#     url = url.rstrip("/").replace(".git", "")
#     if "github.com" in url:
#         parts = url.split("github.com/")[-1].split("/")
#         return "github", parts[0], parts[1]
#     elif "bitbucket.org" in url:
#         parts = url.split("bitbucket.org/")[-1].split("/")
#         return "bitbucket", parts[0], parts[1]
#     else:
#         print("❌ Only github.com and bitbucket.org URLs supported.")
#         sys.exit(1)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--repo", required=True, help="Full repo URL")
#     args = parser.parse_args()

#     if not WEBHOOK_URL or not WEBHOOK_SECRET:
#         print("❌ WEBHOOK_URL and WEBHOOK_SECRET must be set in .env")
#         sys.exit(1)

#     provider, owner, repo = parse_repo_url(args.repo)

#     print(f"Registering webhook for {provider}: {owner}/{repo}")
#     print(f"  → Payload URL : {WEBHOOK_URL}")
#     print(f"  → Secret      : {'*' * len(WEBHOOK_SECRET)}")

#     if provider == "github":
#         if not GITHUB_TOKEN:
#             print("❌ GITHUB_TOKEN not set in .env")
#             sys.exit(1)
#         register_github(owner, repo)
#     else:
#         if not BB_USER or not BB_PASS:
#             print("❌ BITBUCKET_USER and BITBUCKET_APP_PASSWORD not set in .env")
#             sys.exit(1)
#         register_bitbucket(owner, repo)
#!/usr/bin/env python3
"""
setup_webhook.py
----------------
Registers GitHub webhooks automatically.
For Bitbucket: uses manual webhook setup (API tokens DO NOT support webhook creation).

Usage:
    python setup_webhook.py --repo https://github.com/org/myrepo
    python setup_webhook.py --repo https://bitbucket.org/org/myrepo
"""

import os, sys, argparse, requests
from dotenv import load_dotenv

load_dotenv()

WEBHOOK_URL    = os.getenv("WEBHOOK_URL")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN")

# Bitbucket now runs in MANUAL mode only
BITBUCKET_MANUAL_MODE = os.getenv("BITBUCKET_MANUAL_MODE", "true").lower() == "true"


# -----------------------------
# GitHub webhook registration
# -----------------------------
def register_github(owner: str, repo: str):
    url = f"https://api.github.com/repos/{owner}/{repo}/hooks"

    payload = {
        "name": "web",
        "active": True,
        "events": ["push", "pull_request"],
        "config": {
            "url": WEBHOOK_URL,
            "content_type": "json",
            "secret": WEBHOOK_SECRET,
            "insecure_ssl": "0",
        },
    }

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    r = requests.post(url, json=payload, headers=headers)

    if r.status_code == 201:
        print(f"✅ GitHub webhook registered → {WEBHOOK_URL}")
        print(f"   Hook ID: {r.json()['id']}")
    elif r.status_code == 422:
        print("⚠️ GitHub webhook already exists — skipping.")
    else:
        print(f"❌ GitHub error {r.status_code}: {r.text}")
        sys.exit(1)


# -----------------------------
# Bitbucket (MANUAL ONLY)
# -----------------------------
def register_bitbucket(workspace: str, repo: str):
    print("\n⚠️ Bitbucket automatic webhook creation is NOT supported in your environment.\n")

    print("👉 Please add webhook manually:")
    print("-----------------------------------")
    print(f"Repository : {workspace}/{repo}")
    print(f"URL        : {WEBHOOK_URL}")
    print(f"Secret     : {WEBHOOK_SECRET}")
    print("\nTriggers:")
    print("  ✔ Push (REQUIRED)")
    print("  ✔ Pull Request Created (optional)")
    print("  ✔ Pull Request Updated (optional)")
    print("  ✔ Merged (optional)")
    print("\nSettings:")
    print("  Status: Active")
    print("  SSL verify: ON (recommended)")
    print("-----------------------------------\n")

    print("After adding webhook, your pipeline will work normally.\n")


# -----------------------------
# Repo parser
# -----------------------------
def parse_repo_url(url: str):
    url = url.rstrip("/").replace(".git", "")

    if "github.com" in url:
        owner, repo = url.split("github.com/")[-1].split("/")
        return "github", owner, repo

    if "bitbucket.org" in url:
        workspace, repo = url.split("bitbucket.org/")[-1].split("/")
        return "bitbucket", workspace, repo

    print("❌ Only github.com and bitbucket.org URLs supported.")
    sys.exit(1)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()

    if not WEBHOOK_URL or not WEBHOOK_SECRET:
        print("❌ WEBHOOK_URL and WEBHOOK_SECRET must be set in .env")
        sys.exit(1)

    provider, owner, repo = parse_repo_url(args.repo)

    print(f"\nRegistering webhook for {provider}: {owner}/{repo}")
    print(f"→ Payload URL : {WEBHOOK_URL}")
    print(f"→ Secret      : {'*' * len(WEBHOOK_SECRET)}\n")

    if provider == "github":
        if not GITHUB_TOKEN:
            print("❌ GITHUB_TOKEN missing in .env")
            sys.exit(1)
        register_github(owner, repo)

    else:
        # Bitbucket = manual mode only
        register_bitbucket(owner, repo)