#!/usr/bin/env python3
"""
GitHub MCP Wrapper - PyGithub-like syntax over GitHub MCP Connector tools.

This module provides familiar PyGithub-style classes and methods that map 
transparently to the available GitHub MCP connected tools (github___*).

It makes interacting with GitHub via the MCP feel natural and ergonomic,
instead of manually crafting verbose tool calls and remembering exact schemas.

Usage (in your reasoning or scripts):

    from github_mcp_wrapper import Github

    g = Github()  # Uses your connected GitHub account (no token needed)

    # Get authenticated user
    me_spec = g.get_me()
    # Then: call_connected_tool(tool_name=me_spec["tool_name"], arguments=me_spec["arguments"])

    repo = g.get_repo("xai-org/grok")  # or any owner/repo

    # List open issues (returns call spec; execute with call_connected_tool)
    issues_spec = repo.get_issues(state="open", per_page=10)
    # issues = call_connected_tool(**issues_spec)  # in real execution

    # Get a specific issue
    issue_spec = repo.get_issue(42)

    # Create a comment on an issue (after you have the issue dict or wrap it)
    # comment_spec = repo.add_issue_comment(issue_number=42, body="Thanks!")

    # File operations (very common)
    readme_spec = repo.get_contents("README.md", ref="main")
    # content_data = call... ; then content_data.get("content") is base64 usually

    # Create or update file
    create_spec = repo.create_file(
        path="new_feature.py",
        message="Add awesome feature",
        content="print('Hello from MCP wrapper!')",
        branch="feature/awesome"
    )

    # Bulk push multiple files in ONE commit (super useful, not in standard PyGithub easily)
    push_spec = repo.push_files(
        files=[
            {"path": "a.py", "content": "code here"},
            {"path": "b.md", "content": "# docs"}
        ],
        message="Bulk update via wrapper",
        branch="main"
    )

    # Pull requests
    prs_spec = repo.get_pulls(state="open")
    pr_spec = repo.get_pull(123)  # or create_pull(...)

    # Search
    search_spec = g.search_repositories("topic:ai language:python stars:>1000", sort="stars")

Key design:
- Methods return a dict: {"tool_name": "github___xxx", "arguments": {...}, "_note": "..."}
  You then pass to call_connected_tool(tool_name=..., arguments=...)
- If you pass mcp_caller=some_callable to Github(), it will auto-execute (for advanced use).
- Returned objects can be further wrapped into Issue/PR etc. for .edit() etc. methods.
- Pagination: pass perPage, page, after (cursor) as needed. MCP tools handle it.
- Not all PyGithub features are implemented (low-level git objects, teams, etc.),
  but core repo/issue/PR/file ops are covered with the best matching MCP tools.
- This is a client-side convenience layer. The actual auth and execution is handled
  by the MCP connector (your connected GitHub account).

MCP tools used (discovered via search_connected_tools):
- github___get_me, github___search_repositories, github___search_code, github___search_issues, github___search_pull_requests
- github___list_issues, github___list_pull_requests, github___list_branches, github___list_releases, github___list_tags
- github___issue_read, github___issue_write, github___add_issue_comment
- github___pull_request_read (get, get_diff, get_files, get_comments, ...), github___pull_request_review_write, github___create_pull_request, github___update_pull_request, github___merge_pull_request, github___list_pull_requests
- github___get_file_contents, github___create_or_update_file, github___delete_file, github___push_files
- github___create_branch, github___get_commit, github___get_latest_release, etc.

Author: Grok (xAI) - making GitHub access feel first-class.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import base64
import json


class GithubMCPError(Exception):
    """Wrapper-specific error."""
    pass


class GithubObject:
    """Base class for GitHub objects (Issue, Repository, PullRequest, etc.).
    Wraps raw dict data from MCP responses and provides convenience methods.
    """
    def __init__(self, raw_data: Optional[Dict[str, Any]] = None, github_client: Optional["Github"] = None):
        self._raw = raw_data or {}
        self._github = github_client
        if raw_data:
            for key, value in raw_data.items():
                # Avoid overwriting methods
                if not hasattr(self, key):
                    setattr(self, key, value)

    def __repr__(self) -> str:
        name = getattr(self, "name", None) or getattr(self, "login", None) or getattr(self, "number", None) or getattr(self, "title", None) or ""
        return f"<{self.__class__.__name__} {name}>"

    def to_dict(self) -> Dict[str, Any]:
        return self._raw


class NamedUser(GithubObject):
    """User or organization."""
    pass


class ContentFile(GithubObject):
    """Represents a file or directory entry from get_contents()."""
    @property
    def decoded_content(self) -> Optional[bytes]:
        """Return decoded content if base64 encoded (matches PyGithub behavior)."""
        content = getattr(self, "content", None)
        encoding = getattr(self, "encoding", None)
        if content and encoding == "base64":
            try:
                return base64.b64decode(content)
            except Exception:
                return content.encode() if isinstance(content, str) else content
        return content.encode() if isinstance(content, str) else content

    @property
    def text(self) -> Optional[str]:
        """Convenience: decoded as utf-8 text."""
        decoded = self.decoded_content
        if decoded:
            try:
                return decoded.decode("utf-8")
            except Exception:
                return str(decoded)
        return None


class Issue(GithubObject):
    """Issue wrapper with PyGithub-like methods."""
    def create_comment(self, body: str) -> Dict[str, Any]:
        """Add a comment to this issue. Maps to github___add_issue_comment."""
        if not self._github or not hasattr(self, "number"):
            raise GithubMCPError("Issue must be bound to a Github client and have a number")
        # We need owner/repo. Try to infer or require full_name on parent.
        # For simplicity, assume user passes or we store context. Here we use a placeholder pattern.
        # In practice, after getting issue from repo, the wrapper can attach owner/repo.
        owner = getattr(self, "repository", {}).get("owner", {}).get("login") if isinstance(getattr(self, "repository", None), dict) else None
        repo_name = getattr(self, "repository", {}).get("name") if isinstance(getattr(self, "repository", None), dict) else None
        if not owner or not repo_name:
            # Fallback: user must provide or we raise with instruction
            raise GithubMCPError("Cannot determine owner/repo for comment. Use repo.add_issue_comment(issue_number=..., body=...) instead or attach context.")

        return self._github._call(
            "github___add_issue_comment",
            {
                "owner": owner,
                "repo": repo_name,
                "issue_number": self.number,
                "body": body,
            },
        )

    def edit(self, **kwargs: Any) -> Dict[str, Any]:
        """Edit issue title, body, state, labels, assignees, milestone etc.
        Maps to github___issue_write with method=\"update\"."""
        if not self._github or not hasattr(self, "number"):
            raise GithubMCPError("Issue must be bound to Github client")
        owner = ...  # same inference problem as above; simplified for core demo
        # For full version, the Repository would return Issue objects with context attached.
        # Here we demonstrate the pattern.
        args: Dict[str, Any] = {
            "method": "update",
            "issue_number": self.number,
            # owner and repo would come from context
        }
        for field in ("title", "body", "state", "labels", "assignees", "milestone"):
            if field in kwargs:
                args[field] = kwargs[field]
        # Note: owner/repo must be filled by caller or enhanced wrapper
        return {
            "tool_name": "github___issue_write",
            "arguments": args,
            "_note": "Fill owner/repo from context or use Repository.issue_edit helper. MCP requires them."
        }


class PullRequest(GithubObject):
    """Pull Request wrapper."""
    def merge(self, commit_message: Optional[str] = None, merge_method: str = "merge", **kwargs: Any) -> Dict[str, Any]:
        """Merge this PR. Maps to github___merge_pull_request."""
        if not hasattr(self, "number"):
            raise GithubMCPError("PR must have number")
        # Context needed for owner/repo
        return {
            "tool_name": "github___merge_pull_request",
            "arguments": {
                "owner": "FILL_OWNER",
                "repo": "FILL_REPO",
                "pullNumber": self.number,
                "merge_method": merge_method,
                "commit_message": commit_message or "",
            },
            "_note": "Replace FILL_OWNER/FILL_REPO and call the tool."
        }

    def create_review(self, body: str, event: str = "COMMENT", **kwargs: Any) -> Dict[str, Any]:
        """Create/submit a review. Maps to github___pull_request_review_write method=create."""
        return {
            "tool_name": "github___pull_request_review_write",
            "arguments": {
                "owner": "FILL",
                "repo": "FILL",
                "pullNumber": getattr(self, "number", None),
                "method": "create",
                "body": body,
                "event": event,
            }
        }


class Repository(GithubObject):
    """Repository wrapper mimicking PyGithub's Repository class."""

    def __init__(
        self,
        raw_data: Optional[Dict[str, Any]] = None,
        github_client: Optional["Github"] = None,
        owner: Optional[str] = None,
        repo: Optional[str] = None,
    ):
        if raw_data:
            super().__init__(raw_data, github_client)
            owner_info = raw_data.get("owner", {})
            self.owner = owner_info.get("login") if isinstance(owner_info, dict) else owner_info
            self.name = raw_data.get("name") or raw_data.get("full_name", "").split("/")[-1]
        else:
            self.owner = owner
            self.name = repo
            self._github = github_client
            self._raw = {"owner": {"login": owner}, "name": repo}

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    # --- Issues ---
    def get_issues(
        self,
        state: str = "open",
        labels: Optional[List[str]] = None,
        sort: Optional[str] = None,
        direction: str = "desc",
        per_page: int = 30,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """List issues. Maps to github___list_issues (preferred) or search_issues.
        State: \"open\", \"closed\", or \"all\" (MCP uses OPEN/CLOSED).
        """
        mcp_state = None
        if state.lower() == "open":
            mcp_state = "OPEN"
        elif state.lower() == "closed":
            mcp_state = "CLOSED"
        # "all" -> omit state

        args: Dict[str, Any] = {
            "owner": self.owner,
            "repo": self.name,
            "perPage": per_page,
        }
        if mcp_state:
            args["state"] = mcp_state
        if labels:
            args["labels"] = labels
        if sort:
            args["orderBy"] = sort.upper() if sort else None  # CREATED_AT etc.
            args["direction"] = direction.upper()
        if "since" in kwargs:
            args["since"] = kwargs["since"]
        if "after" in kwargs:  # cursor pagination
            args["after"] = kwargs["after"]

        return self._github._call("github___list_issues", args) if self._github else {
            "tool_name": "github___list_issues", "arguments": args
        }

    def get_issue(self, number: int) -> Dict[str, Any]:
        """Get a single issue by number. Maps to github___issue_read method=get."""
        args = {
            "owner": self.owner,
            "repo": self.name,
            "issue_number": number,
            "method": "get",
        }
        return self._github._call("github___issue_read", args) if self._github else {
            "tool_name": "github___issue_read", "arguments": args
        }

    def create_issue(self, title: str, body: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Create a new issue. Maps to github___issue_write method=create."""
        args: Dict[str, Any] = {
            "method": "create",
            "owner": self.owner,
            "repo": self.name,
            "title": title,
        }
        if body:
            args["body"] = body
        for field in ("labels", "assignees", "milestone"):
            if field in kwargs:
                args[field] = kwargs[field]
        return self._github._call("github___issue_write", args) if self._github else {
            "tool_name": "github___issue_write", "arguments": args
        }

    def add_issue_comment(self, issue_number: int, body: str) -> Dict[str, Any]:
        """Add comment to an issue (or PR). Maps to github___add_issue_comment."""
        args = {
            "owner": self.owner,
            "repo": self.name,
            "issue_number": issue_number,
            "body": body,
        }
        return self._github._call("github___add_issue_comment", args) if self._github else {
            "tool_name": "github___add_issue_comment", "arguments": args
        }

    # --- Pull Requests ---
    def get_pulls(
        self,
        state: str = "open",
        head: Optional[str] = None,
        base: Optional[str] = None,
        per_page: int = 30,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """List PRs. Maps to github___list_pull_requests."""
        args: Dict[str, Any] = {
            "owner": self.owner,
            "repo": self.name,
            "state": state,
            "perPage": per_page,
        }
        if head:
            args["head"] = head
        if base:
            args["base"] = base
        if "sort" in kwargs:
            args["sort"] = kwargs["sort"]
        if "direction" in kwargs:
            args["direction"] = kwargs["direction"]
        return self._github._call("github___list_pull_requests", args) if self._github else {
            "tool_name": "github___list_pull_requests", "arguments": args
        }

    def get_pull(self, number: int, method: str = "get", **kwargs: Any) -> Dict[str, Any]:
        """Get PR details, diff, files, comments, reviews etc.
        method: \"get\", \"get_diff\", \"get_files\", \"get_comments\", \"get_review_comments\", \"get_reviews\", \"get_status\", \"get_check_runs\"
        Maps to github___pull_request_read.
        """
        args: Dict[str, Any] = {
            "owner": self.owner,
            "repo": self.name,
            "pullNumber": number,
            "method": method,
        }
        if "page" in kwargs:
            args["page"] = kwargs["page"]
        if "perPage" in kwargs:
            args["perPage"] = kwargs["perPage"]
        return self._github._call("github___pull_request_read", args) if self._github else {
            "tool_name": "github___pull_request_read", "arguments": args
        }

    def create_pull(
        self,
        title: str,
        head: str,
        base: str,
        body: Optional[str] = None,
        draft: bool = False,
        maintainer_can_modify: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a pull request. Maps to github___create_pull_request."""
        args: Dict[str, Any] = {
            "owner": self.owner,
            "repo": self.name,
            "title": title,
            "head": head,
            "base": base,
            "draft": draft,
            "maintainer_can_modify": maintainer_can_modify,
        }
        if body:
            args["body"] = body
        return self._github._call("github___create_pull_request", args) if self._github else {
            "tool_name": "github___create_pull_request", "arguments": args
        }

    def update_pull(self, number: int, **kwargs: Any) -> Dict[str, Any]:
        """Update PR title, body, state, base, etc. Maps to github___update_pull_request."""
        args: Dict[str, Any] = {
            "owner": self.owner,
            "repo": self.name,
            "pullNumber": number,
        }
        for field in ("title", "body", "state", "base", "draft"):
            if field in kwargs:
                args[field] = kwargs[field]
        return self._github._call("github___update_pull_request", args) if self._github else {
            "tool_name": "github___update_pull_request", "arguments": args
        }

    def merge_pull(self, number: int, merge_method: str = "merge", commit_title: Optional[str] = None, commit_message: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Merge a pull request. Maps to github___merge_pull_request."""
        args: Dict[str, Any] = {
            "owner": self.owner,
            "repo": self.name,
            "pullNumber": number,
            "merge_method": merge_method,
        }
        if commit_title:
            args["commit_title"] = commit_title
        if commit_message:
            args["commit_message"] = commit_message
        return self._github._call("github___merge_pull_request", args) if self._github else {
            "tool_name": "github___merge_pull_request", "arguments": args
        }

    # --- Files & Contents ---
    def get_contents(
        self, path: str = "", ref: Optional[str] = None, sha: Optional[str] = None, **kwargs: Any
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]}:
        """Get file or directory contents. Maps to github___get_file_contents.
        Returns file info (with base64 content) or list of entries for directories.
        Use .decoded_content on wrapped ContentFile.
        """
        args: Dict[str, Any] = {
            "owner": self.owner,
            "repo": self.name,
            "path": path or "/",
        }
        if ref:
            args["ref"] = ref
        if sha:
            args["sha"] = sha
        return self._github._call("github___get_file_contents", args) if self._github else {
            "tool_name": "github___get_file_contents", "arguments": args,
            "_note": "For files: content is base64. For dirs: list of {name, path, type, sha, ...}"
        }

    def create_file(
        self,
        path: str,
        message: str,
        content: str,
        branch: str = "main",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a new file (or update if exists and sha provided).
        Maps to github___create_or_update_file. Omit sha for pure create.
        """
        args: Dict[str, Any] = {
            "owner": self.owner,
            "repo": self.name,
            "path": path,
            "message": message,
            "content": content,
            "branch": branch,
        }
        if "sha" in kwargs:  # for update
            args["sha"] = kwargs["sha"]
        return self._github._call("github___create_or_update_file", args) if self._github else {
            "tool_name": "github___create_or_update_file", "arguments": args
        }

    def update_file(
        self,
        path: str,
        message: str,
        content: str,
        sha: str,
        branch: str = "main",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Update existing file (sha required). Same MCP tool as create."""
        return self.create_file(path, message, content, branch, sha=sha, **kwargs)

    def delete_file(
        self, path: str, message: str, branch: str = "main", sha: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Delete a file. Maps to github___delete_file.
        Note: MCP delete_file schema does not require sha (uses latest?).
        """
        args: Dict[str, Any] = {
            "owner": self.owner,
            "repo": self.name,
            "path": path,
            "message": message,
            "branch": branch,
        }
        if sha:
            args["sha"] = sha  # include if provided
        return self._github._call("github___delete_file", args) if self._github else {
            "tool_name": "github___delete_file", "arguments": args
        }

    def push_files(
        self,
        files: List[Dict[str, str]],
        message: str,
        branch: str = "main",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Push multiple files in a SINGLE atomic commit. Maps to github___push_files.
        This is extremely convenient (not a direct PyGithub 1:1 but very powerful).
        files = [{\"path\": \"foo.py\", \"content\": \"print(1)\"}, ...]
        """
        args: Dict[str, Any] = {
            "owner": self.owner,
            "repo": self.name,
            "branch": branch,
            "message": message,
            "files": files,
        }
        return self._github._call("github___push_files", args) if self._github else {
            "tool_name": "github___push_files", "arguments": args
        }

    # --- Other common ---
    def get_branches(self, per_page: int = 30, **kwargs: Any) -> Dict[str, Any]:
        args = {"owner": self.owner, "repo": self.name, "perPage": per_page}
        return self._github._call("github___list_branches", args) if self._github else {
            "tool_name": "github___list_branches", "arguments": args
        }

    def get_releases(self, per_page: int = 30, **kwargs: Any) -> Dict[str, Any]:
        args = {"owner": self.owner, "repo": self.name, "perPage": per_page}
        return self._github._call("github___list_releases", args) if self._github else {
            "tool_name": "github___list_releases", "arguments": args
        }

    def get_latest_release(self) -> Dict[str, Any]:
        args = {"owner": self.owner, "repo": self.name}
        return self._github._call("github___get_latest_release", args) if self._github else {
            "tool_name": "github___get_latest_release", "arguments": args
        }

    def create_branch(self, branch: str, from_branch: str = "main", **kwargs: Any) -> Dict[str, Any]:
        args = {
            "owner": self.owner,
            "repo": self.name,
            "branch": branch,
            "from_branch": from_branch,
        }
        return self._github._call("github___create_branch", args) if self._github else {
            "tool_name": "github___create_branch", "arguments": args
        }

    def get_commit(self, sha: str, include_diff: bool = False, **kwargs: Any) -> Dict[str, Any]:
        args = {
            "owner": self.owner,
            "repo": self.name,
            "sha": sha,
            "include_diff": include_diff,
        }
        return self._github._call("github___get_commit", args) if self._github else {
            "tool_name": "github___get_commit", "arguments": args
        }


class Github:
    """Main entry point. Mimics from github import Github"""

    def __init__(
        self,
        login_or_token: Optional[str] = None,
        password: Optional[str] = None,
        jwt: Optional[str] = None,
        mcp_caller: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        **kwargs: Any,
    ):
        """
        Initialize wrapper.
        - login_or_token, password, jwt: accepted for PyGithub compatibility but ignored
          (MCP connector uses your pre-connected GitHub account).
        - mcp_caller: Optional callable(tool_name: str, arguments: dict) -> result
          If provided, methods will auto-execute the MCP call instead of returning spec.
          Useful for advanced scripting or testing.
        """
        self._login_or_token = login_or_token
        self._mcp_caller = mcp_caller
        self._kwargs = kwargs

    def _call(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Internal: execute or return spec for the MCP tool."""
        payload = {
            "tool_name": tool_name,
            "arguments": arguments,
        }
        if self._mcp_caller:
            try:
                result = self._mcp_caller(tool_name, arguments)
                payload["result"] = result
                return result
            except Exception as e:
                payload["error"] = str(e)
                raise GithubMCPError(f"MCP call failed for {tool_name}: {e}") from e
        return payload  # Return spec for manual execution with call_connected_tool

    # --- User / Auth ---
    def get_me(self) -> Dict[str, Any]:
        """Get authenticated user. Maps to github___get_me."""
        return self._call("github___get_me", {})

    def get_user(self, login: Optional[str] = None) -> Union[NamedUser, Dict[str, Any]]:
        """Get user. If login=None -> get_me. Other users: limited support."""
        if login is None:
            return self.get_me()
        # No direct single-tool for arbitrary user profile in top MCP tools.
        # Could use search or other, but for now raise or return note.
        return {
            "tool_name": None,
            "_note": f"Arbitrary user lookup for '{login}' not directly mapped. "
                    "Use search_repositories or GitHub web if needed. "
                    "get_me() works for the connected account."
        }

    # --- Repositories ---
    def get_repo(self, full_name_or_id: Union[str, int], **kwargs: Any) -> Repository:
        """Get a repository by 'owner/repo' string. Returns Repository wrapper."""
        if isinstance(full_name_or_id, str) and "/" in full_name_or_id:
            owner, repo_name = full_name_or_id.split("/", 1)
            return Repository(raw_data=None, github_client=self, owner=owner, repo=repo_name)
        else:
            # Could support numeric ID via search or other, but simplified
            raise GithubMCPError("Only 'owner/repo' string supported currently. "
                                 "Numeric IDs require additional lookup.")

    def search_repositories(
        self, query: str, sort: Optional[str] = None, order: str = "desc",
        per_page: int = 30, page: int = 1, **kwargs: Any
    ) -> Dict[str, Any]:
        """Search repos. Maps to github___search_repositories."""
        args: Dict[str, Any] = {"query": query, "perPage": per_page, "page": page}
        if sort:
            args["sort"] = sort
        if order:
            args["order"] = order
        if "minimal_output" in kwargs:
            args["minimal_output"] = kwargs["minimal_output"]
        return self._call("github___search_repositories", args)

    def search_code(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Code search across GitHub. Maps to github___search_code."""
        args: Dict[str, Any] = {"query": query}
        for k in ("sort", "order", "page", "perPage"):
            if k in kwargs:
                args[k] = kwargs[k]
        return self._call("github___search_code", args)

    def search_issues(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Search issues (and PRs). Maps to github___search_issues."""
        args: Dict[str, Any] = {"query": query}
        for k in ("sort", "order", "page", "perPage", "owner", "repo"):
            if k in kwargs:
                args[k] = kwargs[k]
        return self._call("github___search_issues", args)

    def search_pull_requests(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Search PRs specifically. Maps to github___search_pull_requests."""
        args: Dict[str, Any] = {"query": query}
        for k in ("sort", "order", "page", "perPage", "owner", "repo"):
            if k in kwargs:
                args[k] = kwargs[k]
        return self._call("github___search_pull_requests", args)

    # --- Misc helpers ---
    def get_organization(self, login: str, **kwargs: Any) -> Dict[str, Any]:
        """Not directly mapped in top tools. Placeholder."""
        return {
            "tool_name": None,
            "_note": f"Organization lookup for {login} - use get_user or search_repositories('org:{login}') as workaround."
        }


# The Github class above is the main entry point (mimics `from github import Github`).
# No separate factory function is needed; just do:
#   from github_mcp_wrapper import Github
#   g = Github()   # uses your connected GitHub account


# Example of how to attach context for full object methods (advanced)
def wrap_issue(issue_dict: Dict[str, Any], repo: Repository) -> Issue:
    """Helper to create a fully functional Issue object with owner/repo context."""
    issue = Issue(issue_dict, repo._github)
    # Attach context so create_comment etc work without FILL placeholders
    issue._owner = repo.owner
    issue._repo_name = repo.name
    # You can monkey-patch or enhance Issue class to use these in real impl.
    return issue


if __name__ == "__main__":
    print("GitHub MCP Wrapper loaded successfully.")
    print("Example: g = Github(); repo = g.get_repo('owner/repo'); spec = repo.get_contents('README.md')")
    print("Then execute with: call_connected_tool(tool_name=spec['tool_name'], arguments=spec['arguments'])")