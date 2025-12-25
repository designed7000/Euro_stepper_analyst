"""
Home page enrichment utilities for NBA Player DNA dashboard.
Includes quiz generation, fun facts, and dynamic content.
"""

import random
import pandas as pd
from datetime import datetime
import hashlib


def generate_daily_seed():
    """Generate a deterministic seed based on today's date for consistent daily content."""
    today = datetime.now().strftime("%Y-%m-%d")
    seed = int(hashlib.md5(today.encode()).hexdigest(), 16)
    return seed


def generate_quiz_question(league_stats, season):
    """
    Generate a random quiz question from league stats.
    Seed is deterministic based on date so same user gets same question all day.
    
    Returns: dict with question, options, correct_answer, explanation
    """
    seed = generate_daily_seed()
    random.seed(seed)
    
    if league_stats.empty:
        return None
    
    question_type = random.choice(['leading_stat', 'player_comparison', 'stat_guess'])
    
    if question_type == 'leading_stat':
        return _generate_leading_stat_question(league_stats)
    elif question_type == 'player_comparison':
        return _generate_player_comparison_question(league_stats)
    else:
        return _generate_stat_guess_question(league_stats)


def _generate_leading_stat_question(league_stats):
    """Who leads the league in X stat?"""
    stats_available = {
        'PTS': ('Points Per Game', 'PTS'),
        'AST': ('Assists Per Game', 'AST'),
        'REB': ('Rebounds Per Game', 'REB'),
        'FG_PCT': ('Field Goal Percentage', 'FG_PCT'),
    }

    stat_key, (stat_label, stat_col) = random.choice(list(stats_available.items()))

    if stat_col not in league_stats.columns:
        return None

    top_player = league_stats.nlargest(1, stat_col).iloc[0]
    correct_answer = top_player['PLAYER_NAME']

    # Get 3 other random players as wrong answers
    others_df = league_stats[league_stats['PLAYER_NAME'] != correct_answer]
    n_wrong = min(3, max(len(others_df), 0))
    wrong_answers = others_df.sample(n_wrong)['PLAYER_NAME'].tolist() if n_wrong > 0 else []

    options = [correct_answer] + wrong_answers
    random.shuffle(options)
    
    return {
        'type': 'multiple_choice',
        'question': f"Who leads the NBA in {stat_label} this season?",
        'options': options,
        'correct_answer': correct_answer,
        'explanation': f"{correct_answer} is averaging {top_player[stat_col]:.1f} {stat_label.lower()} per game.",
        'difficulty': 'Easy'
    }


def _generate_player_comparison_question(league_stats):
    """Compare two players' stats."""
    if len(league_stats) < 2:
        return None
    
    player1, player2 = league_stats.sample(2).itertuples(index=False, name='Player')
    
    stat_col = random.choice(['PTS', 'AST', 'REB', 'TS_PCT'])
    if stat_col not in league_stats.columns:
        stat_col = 'PTS'
    
    stat_names = {
        'PTS': 'Points Per Game',
        'AST': 'Assists Per Game', 
        'REB': 'Rebounds Per Game',
        'TS_PCT': 'True Shooting Percentage'
    }
    
    stat_name = stat_names.get(stat_col, stat_col)
    
    player1_val = getattr(player1, stat_col, 0)
    player2_val = getattr(player2, stat_col, 0)
    
    if player1_val > player2_val:
        correct = player1.PLAYER_NAME
        explanation = f"{player1.PLAYER_NAME} ({player1_val:.1f}) leads {player2.PLAYER_NAME} ({player2_val:.1f}) in {stat_name.lower()}."
    else:
        correct = player2.PLAYER_NAME
        explanation = f"{player2.PLAYER_NAME} ({player2_val:.1f}) leads {player1.PLAYER_NAME} ({player1_val:.1f}) in {stat_name.lower()}."
    
    options = [correct, [p for p in [player1.PLAYER_NAME, player2.PLAYER_NAME] if p != correct][0]]
    
    return {
        'type': 'multiple_choice',
        'question': f"Who has more {stat_name.lower()}: {player1.PLAYER_NAME} or {player2.PLAYER_NAME}?",
        'options': options,
        'correct_answer': correct,
        'explanation': explanation,
        'difficulty': 'Medium'
    }


def _generate_stat_guess_question(league_stats):
    """Guess a player from their stats."""
    if len(league_stats) < 4:
        return None
    
    player = league_stats.sample(1).iloc[0]
    
    stats_to_show = f"PPG: {player.get('PTS', 0):.1f}, APG: {player.get('AST', 0):.1f}, RPG: {player.get('REB', 0):.1f}"
    
    other_players = league_stats[league_stats['PLAYER_NAME'] != player['PLAYER_NAME']].sample(min(3, len(league_stats) - 1))
    wrong_answers = other_players['PLAYER_NAME'].tolist()
    
    options = [player['PLAYER_NAME']] + wrong_answers
    random.shuffle(options)
    
    return {
        'type': 'multiple_choice',
        'question': f"Who has these stats? {stats_to_show}",
        'options': options,
        'correct_answer': player['PLAYER_NAME'],
        'explanation': f"The player is {player['PLAYER_NAME']} from {player.get('TEAM_ABBREVIATION', 'NBA')}.",
        'difficulty': 'Hard'
    }


def get_fun_fact(league_stats, season):
    """Generate a rotating fun fact about the league."""
    seed = generate_daily_seed()
    random.seed(seed)
    
    if league_stats.empty:
        return "NBA is in full swing this season!"
    
    fact_type = random.choice(['scoring_leader', 'efficiency_record', 'team_fact', 'notable_streak'])
    
    if fact_type == 'scoring_leader':
        top_scorer = league_stats.nlargest(1, 'PTS').iloc[0]
        return f"Scoring Leader: {top_scorer['PLAYER_NAME']} is averaging {top_scorer['PTS']:.1f} PPG this season."
    
    elif fact_type == 'efficiency_record':
        if 'TS_PCT' in league_stats.columns:
            most_efficient = league_stats.nlargest(1, 'TS_PCT').iloc[0]
            return f"Efficiency Leader: {most_efficient['PLAYER_NAME']} has the highest TS% at {most_efficient['TS_PCT']:.1%}."
        else:
            return "Keep an eye on the most efficient scorers this season!"
    
    elif fact_type == 'team_fact':
        return f"The {season} NBA season is in full swing with amazing performances across the league!"
    
    else:
        return "Check out the latest stats and trends on the platform!"
