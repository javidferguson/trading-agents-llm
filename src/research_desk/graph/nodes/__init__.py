"""One module per node. Bodies stay framework-free -- no langgraph, no langfuse.

A node reads ``state``, reads ``ctx``, and returns a partial-state patch. That
constraint is what makes a node unit-testable in isolation, replayable against
recorded responses, and portable if LangGraph is ever un-adopted.
"""
