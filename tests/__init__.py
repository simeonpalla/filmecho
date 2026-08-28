"""Filmecho agents package.

Deliberately does NOT eagerly import every agent here. An earlier version
did, which meant importing anything under agents.* — even agents.schemas,
which has zero external dependencies — forced google-adk to be installed
and every agent's API key env var to be set, just to import a Pydantic
model. That made the package impossible to unit test cleanly and is a
real fragility problem independent of testing, so it's fixed here: import
each agent module directly (e.g. `from agents.web_sentiment_agent import
web_sentiment_agent`) rather than through this package's namespace.

For `adk web` / `adk run` CLI discovery, that requires a `root_agent`
variable inside a dedicated folder's `agent.py` (ADK's own convention),
not a re-export from here, see the README's testing section for the
wrapper-folder pattern.
"""

