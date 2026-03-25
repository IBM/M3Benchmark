name: "Agent Submission"
description: "Submit a new agent package (JSON files in ZIP format)"
title: "[Agent Submission]: <agent-name>"
labels: ["agent-submission"]
body:
  - type: input
    id: agent_name
    attributes:
      label: Agent Name
      description: Unique name of the agent
      placeholder: e.g. langgraph-react-agent
    validations:
      required: true

  - type: input
    id: maintainer
    attributes:
      label: Maintainer
      description: Name or team responsible for the agent
      placeholder: e.g. John Doe / AI Platform Team
    validations:
      required: true

  - type: input
    id: contact
    attributes:
      label: Maintainer Contact
      description: Email or Slack/Discord handle
      placeholder: e.g. john@company.com
    validations:
      required: true

  - type: input
    id: model
    attributes:
      label: Model Used
      description: Primary LLM(s) used by the agent
      placeholder: e.g. gpt-4.1 / llama-405b
    validations:
      required: true

  - type: dropdown
    id: agent_type
    attributes:
      label: Agent Type
      options:
        - Single-turn
        - Multi-turn
        - Multi-agent
        - Tool-using
        - Retrieval-Augmented (RAG)
    validations:
      required: true

  - type: textarea
    id: description
    attributes:
      label: Agent Description
      description: What does this agent do? Include capabilities and use cases.
      placeholder: |
        This agent performs...
    validations:
      required: true

  - type: textarea
    id: metadata
    attributes:
      label: Metadata (JSON)
      description: Any additional structured metadata
      placeholder: |
        {
          "framework": "LangGraph",
          "tools": ["search", "calculator"],
          "latency_ms": 1200
        }

  - type: input
    id: zip_link
    attributes:
      label: ZIP File Link
      description: Public link to ZIP containing JSON files
      placeholder: e.g. https://github.com/org/repo/releases/download/v1/agent.zip
    validations:
      required: true

  - type: textarea
    id: contents
    attributes:
      label: ZIP Contents Description
      description: Describe what’s inside the ZIP
      placeholder: |
        - inputs.json
        - outputs.json
        - config.json

  - type: checkboxes
    id: validation
    attributes:
      label: Validation Checklist
      options:
        - label: JSON files are valid and well-formed
          required: true
        - label: ZIP file is accessible via the provided link
          required: true
        - label: No sensitive or PII data included
          required: true
        - label: Agent has been tested locally
          required: true

  - type: textarea
    id: additional_notes
    attributes:
      label: Additional Notes
      description: Anything reviewers should know
