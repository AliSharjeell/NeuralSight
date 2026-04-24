---
name: windows-mcp-tool-tester
description: >
  Automated testing skill for Windows-MCP tools. Use this skill whenever the user wants to test,
  validate, benchmark, or evaluate any Windows-MCP tool (App, PowerShell, Screenshot, Snapshot,
  Click, Type, Scroll, Move, Shortcut, Wait, MultiSelect, MultiEdit, Clipboard, Process,
  Notification, FileSystem, Registry, Scrape). Triggers on phrases like "test the Click tool",
  "benchmark Screenshot", "validate FileSystem", "run QA on Registry", "check if PowerShell works",
  "evaluate tool performance", or any mention of testing/validating a Windows-MCP tool.
  Each invocation tests exactly ONE tool.
---

# Windows-MCP Tool Tester

An automated testing skill that generates comprehensive test cases for a single Windows-MCP tool,
executes them, and produces a structured test report with pass/fail results, performance metrics,
and actionable recommendations.

## Available Tools

The following tools are provided by the `windows-mcp` MCP server:

| Tool | Description |
|------|-------------|
| **App** | Launch, resize, or switch Windows applications |
| **Click** | Click screen elements by coordinates or UI label |
| **Type** | Type text into focused element |
| **Shortcut** | Press keyboard shortcuts and key combos |
| **Screenshot** | Capture current screen state as image |
| **Snapshot** | Get UI element tree with interactive element labels |
| **Scroll** | Scroll up/down/left/right |
| **Move** | Move mouse cursor to coordinates, optional drag |
| **PowerShell** | Execute PowerShell commands |
| **Wait** | Pause for N seconds |
| **MultiSelect** | Select multiple items |
| **MultiEdit** | Edit multiple fields |
| **Clipboard** | Get/set clipboard content |
| **Process** | List/kill processes |
| **Notification** | Send Windows toast notifications |
| **FileSystem** | Read/write/copy/move/delete/list/search files |
| **Registry** | Get/set/delete/list registry keys |
| **Scrape** | Extract content from URLs |

## Common Workflows

### Open a browser and navigate
```
1. App(action="launch", app_name="chrome")
2. Wait(seconds=2)
3. Shortcut(keys="ctrl+l")
4. Type(text="https://youtube.com", press_enter=true)
5. Wait(seconds=2)
6. Screenshot() to verify
```

### Search on a website
```
1. Navigate to site (above workflow)
2. Snapshot() to find search box label
3. Click(label="search_box_id")
4. Type(text="search query", press_enter=true)
5. Wait(seconds=2)
6. Screenshot() to verify results
```

### Open an application
```
1. App(action="launch", app_name="notepad")
2. Wait(seconds=1)
3. Screenshot() to verify
```
