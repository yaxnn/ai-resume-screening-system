import pytest
import responses
import requests
from src.github_client import fetch_github_data, extract_username, GithubData, _GITHUB_CACHE

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the in-memory cache before each test."""
    _GITHUB_CACHE.clear()

def test_extract_username():
    assert extract_username("https://github.com/torvalds") == "torvalds"
    assert extract_username("github.com/yaseen-test/") == "yaseen-test"
    assert extract_username("https://github.com/pulls") is None
    assert extract_username("https://github.com/issues/123") is None
    assert extract_username(None) is None
    assert extract_username("https://gitlab.com/torvalds") is None

@responses.activate
def test_fetch_github_data_success():
    username = "testuser"
    url = f"https://github.com/{username}"
    
    # Mock repos
    responses.add(
        responses.GET,
        f"https://api.github.com/users/{username}/repos",
        json=[
            {"name": "repo1", "language": "Python"},
            {"name": "repo2", "language": "Java"}
        ],
        status=200
    )
    
    # Mock events
    responses.add(
        responses.GET,
        f"https://api.github.com/users/{username}/events/public",
        json=[
            {"type": "PushEvent", "repo": {"name": "testuser/repo1"}},
            {"type": "WatchEvent", "repo": {"name": "other/repo"}}
        ],
        status=200
    )
    
    data = fetch_github_data(url)
    assert data.username == "testuser"
    assert data.valid_profile is True
    assert data.error is None
    assert len(data.repos) == 2
    assert data.repos[0]["language"] == "Python"
    assert len(data.events) == 2
    assert data.events[0]["type"] == "PushEvent"
    
    # Test cache hit
    # Even if we don't mock it again, responses will complain if called.
    # Since cache hits return immediately, it won't hit responses.
    data_cached = fetch_github_data(url)
    assert data_cached is data

@responses.activate
def test_fetch_github_data_404():
    username = "notfounduser"
    url = f"https://github.com/{username}"
    
    responses.add(
        responses.GET,
        f"https://api.github.com/users/{username}/repos",
        json={"message": "Not Found"},
        status=404
    )
    
    data = fetch_github_data(url)
    assert data.username == "notfounduser"
    assert data.valid_profile is False
    assert data.error == "Profile not found"
    assert len(data.repos) == 0

@responses.activate
def test_fetch_github_data_403_rate_limit():
    username = "ratelimited"
    url = f"https://github.com/{username}"
    
    responses.add(
        responses.GET,
        f"https://api.github.com/users/{username}/repos",
        json={"message": "API rate limit exceeded"},
        status=403
    )
    
    data = fetch_github_data(url)
    assert data.username == "ratelimited"
    assert data.valid_profile is False
    assert data.error == "API rate limit exceeded or forbidden"

@responses.activate
def test_fetch_github_data_network_error():
    username = "networkerror"
    url = f"https://github.com/{username}"
    
    # ConnectionError simulation
    responses.add(
        responses.GET,
        f"https://api.github.com/users/{username}/repos",
        body=requests.exceptions.ConnectionError("Connection refused")
    )
    
    data = fetch_github_data(url)
    assert data.username == "networkerror"
    assert data.valid_profile is False
    assert "Network or API error" in data.error

def test_fetch_github_data_invalid_url():
    data = fetch_github_data("not a url")
    assert data.username == ""
    assert data.valid_profile is False
    assert data.error == "Invalid or missing GitHub URL"

