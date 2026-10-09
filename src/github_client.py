"""
github_client.py — GitHub API Integration .

Fetches public repository and event data for candidates using the unauthenticated
or authenticated public REST API. Implements in-run caching to avoid duplicate
calls, and includes timeouts and error handling.
"""

import os
import re
import time
import logging
import requests
from dataclasses import dataclass, field
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

@dataclass
class GithubData:
    username: str
    valid_profile: bool = False
    error: Optional[str] = None
    repos: List[Dict] = field(default_factory=list)
    events: List[Dict] = field(default_factory=list)

# In-memory cache for a single run
_GITHUB_CACHE: Dict[str, GithubData] = {}

def extract_username(url: Optional[str]) -> Optional[str]:
    """Extract GitHub username from a URL."""
    if not url:
        return None
    # Match github.com/username, ignoring trailing slashes or repo paths
    # Usernames may only contain alphanumeric characters or single hyphens
    # and cannot begin or end with a hyphen. Max length 39.
    match = re.search(r"github\.com/([a-zA-Z0-9-]+)", url, re.IGNORECASE)
    if match:
        username = match.group(1)
        # Filter out common false positives if people link to github.com/pulls etc.
        if username.lower() not in ["pulls", "issues", "marketplace", "explore", "trending", "topics"]:
            return username
    return None

def fetch_github_data(url: Optional[str]) -> GithubData:
    """
    Fetch public repositories and events for a GitHub URL.
    Returns a GithubData object containing the results.
    """
    username = extract_username(url)
    if not username:
        return GithubData(username="", error="Invalid or missing GitHub URL")

    # Normalize cache key
    cache_key = username.lower()
    if cache_key in _GITHUB_CACHE:
        logger.debug(f"Cache hit for GitHub user: {username}")
        return _GITHUB_CACHE[cache_key]

    data = GithubData(username=username)
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    session = requests.Session()
    session.headers.update(headers)

    try:
        # 1. Fetch Repos (sort by updated, max 10)
        repos_url = f"https://api.github.com/users/{username}/repos"
        resp_repos = session.get(
            repos_url,
            params={"sort": "updated", "per_page": 10},
            timeout=5
        )

        if resp_repos.status_code == 404:
            data.error = "Profile not found"
            _GITHUB_CACHE[cache_key] = data
            return data
        elif resp_repos.status_code == 403:
            # Rate limit or blocked
            data.error = "API rate limit exceeded or forbidden"
            _GITHUB_CACHE[cache_key] = data
            return data
        
        resp_repos.raise_for_status()
        data.valid_profile = True
        repos_json = resp_repos.json()
        
        for repo in repos_json:
            # Keep only relevant fields to save memory
            data.repos.append({
                "name": repo.get("name"),
                "description": repo.get("description"),
                "language": repo.get("language"),
                "updated_at": repo.get("updated_at"),
                "stargazers_count": repo.get("stargazers_count")
            })

        # 2. Fetch Events (public activity)
        events_url = f"https://api.github.com/users/{username}/events/public"
        resp_events = session.get(
            events_url,
            params={"per_page": 10},
            timeout=5
        )

        if resp_events.status_code == 200:
            events_json = resp_events.json()
            for event in events_json:
                data.events.append({
                    "type": event.get("type"),
                    "repo": event.get("repo", {}).get("name"),
                    "created_at": event.get("created_at")
                })
        elif resp_events.status_code == 403:
            logger.warning(f"Rate limited while fetching events for {username}")
            # Do not completely fail, we still have repo data

    except requests.exceptions.RequestException as e:
        logger.warning(f"GitHub API error for {username}: {e}")
        data.error = f"Network or API error: {e}"
        # If we failed early, valid_profile might still be False

    _GITHUB_CACHE[cache_key] = data
    return data

