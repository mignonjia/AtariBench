"""Prompt templates for the game end condition."""

MAX_TIME_PROMPT: str = """
Time Budget:
You are optimizing total reward over a fixed budget of {MAX_TIME} seconds.
Losing a life does not directly reduce your score, but it does consume time
because the game pauses and restarts the player state.
"""
