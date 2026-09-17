"""Chips agent: reasons about the fixture CALENDAR SHAPE, not individual
players - when (if at all) to play Wildcard, Bench Boost, Triple Captain or
Free Hit.

Unlike every other specialist, this agent never populates `recommendations`
or `vetoes` - it isn't arguing for players, it's arguing for TIMING. See
shared/contracts.py's `ChipRecommendation` for its dedicated output field.

"No chip this week" is the expected, correct outcome most of the time - see
scoring.py's module docstring for why that's not an edge case to apologize
for.
"""
