"""Tests for entity linking (see entity_linking.py's module docstring for
the ambiguous-surname rule). Pure function, synthetic player list - no DB.
"""

from __future__ import annotations

from agents.news.entity_linking import Player, link_players


def test_unique_surname_resolves_to_the_right_player_id():
    players = [
        Player(player_id=1, web_name="Salah", team_name="Liverpool"),
        Player(player_id=2, web_name="Haaland", team_name="Man City"),
    ]
    result = link_players("Salah is in fine form ahead of the weekend fixture.", players)
    assert result.player_ids == [1]
    assert result.ambiguous == []


def test_ambiguous_surname_resolved_by_team_mention():
    # Two different players share the surname "Silva" - the documented rule
    # is: disambiguate by team name if the text names exactly one of them.
    players = [
        Player(player_id=10, web_name="Silva", team_name="Fulham"),
        Player(player_id=11, web_name="Silva", team_name="Man City"),
    ]
    result = link_players("Silva impressed for Man City in their win on Saturday.", players)
    assert result.player_ids == [11]
    assert result.ambiguous == []


def test_ambiguous_surname_with_no_disambiguating_team_is_left_unresolved():
    players = [
        Player(player_id=10, web_name="Silva", team_name="Fulham"),
        Player(player_id=11, web_name="Silva", team_name="Man City"),
    ]
    result = link_players("Silva scored twice at the weekend.", players)
    assert result.player_ids == []
    assert len(result.ambiguous) == 1
    surname, candidates = result.ambiguous[0]
    assert surname == "silva"
    assert sorted(candidates) == [10, 11]


def test_no_mention_links_nothing():
    players = [Player(player_id=1, web_name="Salah", team_name="Liverpool")]
    result = link_players("A quiet gameweek with no major news.", players)
    assert result.player_ids == []
    assert result.ambiguous == []
