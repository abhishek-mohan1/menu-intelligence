import pandas as pd

def calculate_trends(df, reliable_items):
    """Calculate month-over-month trends per item"""
    
    df['date']  = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.to_period('M')
    
    def performance_score(group):
        positive = (group['sentiment'] == 'Positive').sum()
        negative = (group['sentiment'] == 'Negative').sum()
        total    = len(group)
        return round((positive - negative) / total, 2)
    
    trend_df = df.groupby(['month', 'menu_item']).apply(performance_score).reset_index()
    trend_df.columns = ['month', 'menu_item', 'score']
    trend_df['month'] = trend_df['month'].astype(str)
    trend_df = trend_df[trend_df['menu_item'].isin(reliable_items)]
    
    # Rising vs Falling
    all_months  = sorted(trend_df['month'].unique())
    mid         = len(all_months) // 2
    first_half  = all_months[:mid]
    second_half = all_months[mid:]
    
    first_avg  = trend_df[trend_df['month'].isin(first_half)].groupby('menu_item')['score'].mean()
    second_avg = trend_df[trend_df['month'].isin(second_half)].groupby('menu_item')['score'].mean()
    
    summary = pd.DataFrame({'first_half': first_avg, 'second_half': second_avg}).dropna()
    summary['change']    = (summary['second_half'] - summary['first_half']).round(2)
    summary['direction'] = summary['change'].apply(
        lambda x: '📈 Rising' if x > 0.05 else ('📉 Falling' if x < -0.05 else '➡️ Stable')
    )
    
    return trend_df, summary