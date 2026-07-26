"""Start the MLflow AgentServer used locally and by Databricks Apps."""

from mlflow.genai.agent_server import AgentServer

from app import endpoint  # Importing registers the decorated endpoint.

agent_server = AgentServer("ResponsesAgent")
app = agent_server.app
app.router.on_startup.append(endpoint.initialize_application)
app.router.on_shutdown.append(endpoint.close_application)


def main() -> None:
    # Fail before opening the HTTP listener if deployment configuration could
    # otherwise resolve a mutable prompt alias or an invalid version.
    endpoint.required_prompt_version()
    agent_server.run(app_import_string="start_server:app")


if __name__ == "__main__":
    main()
