# Speaking Bot

Deploy AI-powered meeting agents that can join and participate in Zoom and Microsoft Teams (and soon on Teams!). These agents have distinct personalities and can engage in conversations based on predefined personas defined in Markdown files.

## Overview

The Meeting Agent Bot allows you to:

-   Launch one or more AI agents into Google Meet or Microsoft Teams (Zoom is due ASAP)
-   Give each agent a unique personality and conversation style
-   Run multiple instances locally or scale to web deployment

## Technical Stack

This bot utilizes:

-   MeetingBaas's APIs for meeting interactions
-   Pipecat's `WebsocketServerTransport` for real-time communication
-   Ngrok for local server exposure

### Multiple Instance Architecture

When running multiple bot instances:

-   Each bot requires a unique public Ngrok URL
-   MeetingBaas communicates with each bot through its dedicated WebSocket
-   Pipecat handles the real-time message routing

**Current Limitations**

Currently, the app only supports 2 simultaneous agents in meetings, limited by local development and ngrok.

**Running Meeting Agents**

To run 1 or 2 meeting agents in a meeting, execute the following commands:
