def extract_menu_items(df, menu_keywords):
    """Extract menu items from processed review text"""
    
    def find_items(text):
        text = str(text).lower()
        found = [item for item in menu_keywords if item in text]
        return ', '.join(found) if found else 'other'
    
    df['menu_item'] = df['processed_text'].apply(find_items)
    
    # Expand rows so each item gets its own row
    df['menu_item'] = df['menu_item'].str.split(', ')
    df = df.explode('menu_item').reset_index(drop=True)
    df = df[df['menu_item'] != 'other']
    
    return df