"""Prompt templates for the game end condition."""

MAX_LOST_LIVES_PROMPT: str = """
Game End Condition:
We control the game length by only allowing {MAX_LOST_LIVES} lives.
So you will lose the game if you lose {MAX_LOST_LIVES} lives.
"""

MAX_LOST_REWARDS_PROMPT: str = """
Game End Condition:
We control the game length by only allowing {MAX_LOST_REWARDS} loss of scores.
So you will lose the game if you lose {MAX_LOST_REWARDS} scores.
"""

MAX_TIME_PROMPT: str = """
Game End Condition:
Note that the game will last for at most {MAX_TIME} seconds.
"""
