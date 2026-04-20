import pandas as pd

def calculate_scores(df, min_mentions=5):
    """Calculate Menu Performance Score per item"""
    
    def performance_score(group):
        positive = (group['sentiment'] == 'Positive').sum()
        negative = (group['sentiment'] == 'Negative').sum()
        total    = len(group)
        return round((positive - negative) / total, 2)
    
    scores = df.groupby('menu_item').apply(performance_score).reset_index()
    scores.columns = ['menu_item', 'performance_score']
    
    counts = df['menu_item'].value_counts().reset_index()
    counts.columns = ['menu_item', 'total_mentions']
    
    scores = scores.merge(counts, on='menu_item')
    scores = scores[scores['total_mentions'] >= min_mentions]
    scores = scores.sort_values('performance_score', ascending=False).reset_index(drop=True)
    
    def assign_band(score):
        if score > 0.6:
            return '🟢 High'
        elif score >= 0.2:
            return '🟡 Moderate'
        else:
            return '🔴 Poor'
    
    def recommend_action(band):
        if 'High' in band:
            return '⭐ Promote'
        elif 'Moderate' in band:
            return '🔧 Rework'
        else:
            return '❌ Drop'
    
    scores['band']   = scores['performance_score'].apply(assign_band)
    scores['action'] = scores['band'].apply(recommend_action)
    
    return scores