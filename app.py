import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# Fix path so Python finds the pipeline folder correctly
pipeline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline')
sys.path.insert(0, pipeline_path)

import extractor
import sentiment
import scorer
import trend

# ── Page config ──────────────────────────────────────────
st.set_page_config(page_title="Menu Intelligence Dashboard",
                   page_icon="🍽️", layout="wide")

st.title("🍽️ Menu Intelligence Dashboard")
st.markdown("*Upload monthly reviews and get instant menu performance insights*")
st.divider()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    restaurant_name = st.text_input("Restaurant Name", "My Restaurant")
    min_mentions    = st.slider("Minimum mentions to include item", 2, 20, 5)

    st.markdown("---")
    st.markdown("**Menu Keywords to Track**")
    default_keywords = "biryani, chicken, millet, vegan, burger, fries, cheese, paneer, rice, naan, kebab, wrap, pizza, pasta, salad, smoothie, juice, coffee, dessert, cake"
    keywords_input = st.text_area("Edit keywords (comma separated)", default_keywords)
    menu_keywords  = [k.strip() for k in keywords_input.split(',')]

# ── File Upload ───────────────────────────────────────────
st.header("📁 Upload Reviews")
uploaded_file = st.file_uploader(
    "Upload your reviews file (CSV or Excel) with 'date' and 'text' columns",
    type=['csv', 'xlsx', 'xls']
)

if uploaded_file:

    file_name = uploaded_file.name

    if file_name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        df = pd.read_excel(uploaded_file)
        total_reviews = len(df)
    st.success(f"✅ Loaded {total_reviews} reviews for **{restaurant_name}**")

    with st.expander("Preview raw data"):
        st.dataframe(df.head())

    # ── Run Pipeline ──────────────────────────────────────
    with st.spinner("Running analysis pipeline... this may take a minute ⏳"):

        # Step 0 - Preprocess raw text if needed
        if 'processed_text' not in df.columns:
            st.info("⚙️ Raw reviews detected — running preprocessing first...")
            import preprocessor
            df = preprocessor.preprocess(df)

        # Step 1 - Extract menu items
        df = extractor.extract_menu_items(df, menu_keywords)

        # Step 2 - Score sentiment if not already present
        if 'sentiment' not in df.columns:
            df = sentiment.score_sentiment(df)

        # Step 3 - Calculate scores and bands
        scores = scorer.calculate_scores(df, min_mentions)

        # Step 4 - Calculate trends
        reliable_items    = scores['menu_item'].tolist()
        trend_df, summary = trend.calculate_trends(df, reliable_items)

    st.success("✅ Analysis complete!")
    st.divider()

    # ── Summary Cards ─────────────────────────────────────
    st.header(f"📊 Results for {restaurant_name}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews",    total_reviews)
    col2.metric("Items Tracked",    len(scores))
    col3.metric("High Performers",  len(scores[scores['band'].str.contains('High')]))
    col4.metric("Needs Attention",  len(scores[scores['band'].str.contains('Moderate|Poor')]))

    st.divider()

    # ── Performance Table ─────────────────────────────────
    st.header("🏆 Menu Performance Scores")
    st.dataframe(scores, use_container_width=True)

    st.divider()

    # ── Bar Chart ─────────────────────────────────────────
    st.header("📊 Performance Score Chart")

    color_map = {'🟢 High': 'green', '🟡 Moderate': 'orange', '🔴 Poor': 'red'}

    fig = px.bar(
        scores,
        x='menu_item',
        y='performance_score',
        color='band',
        color_discrete_map=color_map,
        text='performance_score',
        title="Menu Item Performance Score"
    )
    fig.add_hline(y=0.6, line_dash="dash", line_color="green",
                  annotation_text="High threshold (0.6)")
    fig.add_hline(y=0.2, line_dash="dash", line_color="orange",
                  annotation_text="Moderate threshold (0.2)")
    fig.update_layout(xaxis_title="Menu Item",
                      yaxis_title="Performance Score")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Trend Chart ───────────────────────────────────────
    st.header("📈 Trend Over Time")

    if len(reliable_items) > 0:
        selected_items = st.multiselect(
            "Select items to compare on trend chart",
            options=reliable_items,
            default=reliable_items[:5] if len(reliable_items) >= 5 else reliable_items
        )

        if selected_items:
            trend_filtered = trend_df[trend_df['menu_item'].isin(selected_items)]
            fig2 = px.line(
                trend_filtered,
                x='month',
                y='score',
                color='menu_item',
                markers=True,
                title="Menu Item Performance Trend Over Time"
            )
            fig2.add_hline(y=0.6, line_dash="dash", line_color="green",
                           annotation_text="High threshold")
            fig2.add_hline(y=0.2, line_dash="dash", line_color="orange",
                           annotation_text="Moderate threshold")
            fig2.update_layout(xaxis_title="Month",
                               yaxis_title="Performance Score")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Please select at least one item from the dropdown above.")
    else:
        st.warning("Not enough data to show trends.")

    st.divider()

    # ── Rising vs Falling ─────────────────────────────────
    st.header("🔍 Rising vs Falling Items")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### 📈 Rising")
        rising = summary[summary['direction'] == '📈 Rising']
        if len(rising) > 0:
            for item in rising.index:
                st.success(f"**{item}** (+{rising.loc[item, 'change']})")
        else:
            st.info("No rising items found.")

    with col_b:
        st.markdown("### ➡️ Stable")
        stable = summary[summary['direction'] == '➡️ Stable']
        if len(stable) > 0:
            for item in stable.index:
                st.info(f"**{item}**")
        else:
            st.info("No stable items found.")

    with col_c:
        st.markdown("### 📉 Falling")
        falling = summary[summary['direction'] == '📉 Falling']
        if len(falling) > 0:
            for item in falling.index:
                st.error(f"**{item}** ({falling.loc[item, 'change']})")
        else:
            st.success("No falling items — great!")

    st.divider()

    # ── Business Recommendations ──────────────────────────
    st.header("💡 Business Recommendations")

    high_items     = scores[scores['band'].str.contains('High')]['menu_item'].tolist()
    moderate_items = scores[scores['band'].str.contains('Moderate')]['menu_item'].tolist()
    poor_items     = scores[scores['band'].str.contains('Poor')]['menu_item'].tolist()

    if high_items:
        st.success(f"⭐ **Promote these items** — customers love them: {', '.join(high_items)}")

    if moderate_items:
        st.warning(f"🔧 **Rework these items** — good potential but needs improvement: {', '.join(moderate_items)}")

    if poor_items:
        st.error(f"❌ **Consider dropping or replacing** — consistently poor feedback: {', '.join(poor_items)}")

    # Trend-based extra recommendations
    if len(rising) > 0:
        rising_list = rising.index.tolist()
        st.info(f"📈 **Trending up** — consider featuring these on your menu front: {', '.join(rising_list)}")

    if len(falling) > 0:
        falling_list = falling.index.tolist()
        st.warning(f"📉 **Losing steam** — investigate quality consistency for: {', '.join(falling_list)}")

    st.divider()

    # ── Export ────────────────────────────────────────────
    st.header("⬇️ Export Results")

    col_d, col_e, col_f = st.columns(3)

    with col_d:
        st.download_button(
            label="📥 Download Performance Scores",
            data=scores.to_csv(index=False),
            file_name="menu_performance_scores.csv",
            mime="text/csv"
        )

    with col_e:
        st.download_button(
            label="📥 Download Trend Data",
            data=trend_df.to_csv(index=False),
            file_name="trend_data.csv",
            mime="text/csv"
        )

    with col_f:
        st.download_button(
            label="📥 Download Trend Summary",
            data=summary.to_csv(),
            file_name="trend_summary.csv",
            mime="text/csv"
        )

else:
    # ── Landing screen when no file uploaded ──────────────
    st.info("👆 Please upload a CSV file from the section above to get started.")

    st.markdown("### 📋 How It Works")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**1️⃣ Upload**")
        st.markdown("Upload your monthly reviews CSV file")

    with col2:
        st.markdown("**2️⃣ Extract**")
        st.markdown("System finds menu items in each review")

    with col3:
        st.markdown("**3️⃣ Score**")
        st.markdown("Calculates performance score per item")

    with col4:
        st.markdown("**4️⃣ Act**")
        st.markdown("Get promote / rework / drop recommendations")

    st.markdown("---")
    st.markdown("**Expected CSV format:**")
    st.code("date, processed_text, sentiment\n2024-01-15, biryani chicken great, Positive\n2024-01-18, burger was cold dry, Negative")