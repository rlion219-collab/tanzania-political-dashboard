import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Tanzania Political AI Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("tanzania_swahili_political_trustscore_explained.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data()

st.title("Tanzania Political Sentiment & Trust Score Dashboard")
st.write("Real-time AI analysis for users who do not speak Swahili.")

# --- Sidebar Filters ---
st.sidebar.header("Filter Options")

topic_filter = st.sidebar.multiselect(
    "Select Topic",
    options=df["topic"].unique(),
    default=df["topic"].unique()
)

min_trust_score = st.sidebar.slider(
    "Minimum Trust Score",
    min_value=0,
    max_value=100,
    value=0
)

min_sentiment_confidence = st.sidebar.slider(
    "Minimum Sentiment Confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.0, step=0.05
)

# Apply filters
df_filtered = df[
    (df["topic"].isin(topic_filter)) &
    (df["trust_score"] >= min_trust_score) &
    (df["predicted_confidence"] >= min_sentiment_confidence)
]

# Check if filtered data is empty
if df_filtered.empty:
    st.warning("No data available based on the current filter settings.")
    st.stop()

# --- KPIs ---
st.subheader("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Posts", df_filtered.shape[0])
with col2:
    st.metric("Avg Trust Score", f"{df_filtered['trust_score'].mean():.2f}")
with col3:
    st.metric("Avg Sentiment Confidence", f"{df_filtered['predicted_confidence'].mean():.2f}")
with col4:
    most_frequent_sentiment = df_filtered["predicted_sentiment"].mode()[0] if not df_filtered["predicted_sentiment"].empty else "N/A"
    st.metric("Most Frequent Sentiment", most_frequent_sentiment)


# --- Visualizations ---
st.subheader("Data Visualizations")

# Sentiment Distribution (Pie Chart)
fig_sentiment = px.pie(df_filtered, names="predicted_sentiment", title="Sentiment Distribution")
st.plotly_chart(fig_sentiment, use_container_width=True)

# Trust Score Distribution (Histogram)
fig_trust = px.histogram(df_filtered, x="trust_score", nbins=20, title="Trust Score Distribution")
st.plotly_chart(fig_trust, use_container_width=True)

# Sentiment Over Time (Line Chart)
df_resampled = df_filtered.set_index('timestamp').resample('D').agg({
    'predicted_sentiment': lambda x: x.mode()[0] if not x.empty else 'neutral',
    'trust_score': 'mean'
}).reset_index()

# Map sentiment to numerical values for plotting
sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
df_resampled['sentiment_value'] = df_resampled['predicted_sentiment'].map(sentiment_map)

fig_time = px.line(df_resampled, x='timestamp', y='sentiment_value', title='Sentiment Trend Over Time',
                   labels={'sentiment_value': 'Sentiment (1=Pos, 0=Neu, -1=Neg)'})
st.plotly_chart(fig_time, use_container_width=True)

# --- Topic Insights Table ---
st.subheader("Topic Insights")
topic_summary = df_filtered.groupby("topic").agg(
    total_posts=("tweet_id", "count"),
    avg_trust_score=("trust_score", "mean"),
    avg_sentiment_confidence=("predicted_confidence", "mean"),
    most_frequent_sentiment=("predicted_sentiment", lambda x: x.mode()[0] if not x.empty else "N/A")
).reset_index().round(2)

st.dataframe(topic_summary, use_container_width=True)

# --- Explainability Section ---
st.subheader("Explainability for Individual Tweets")

selected_tweet_id = st.selectbox(
    "Select a Tweet ID to see its details and explanation",
    options=df_filtered["tweet_id"].unique()
)

if selected_tweet_id:
    selected_row = df_filtered[df_filtered["tweet_id"] == selected_tweet_id].iloc[0]
    st.write(f"**Tweet Text:** {selected_row['text']}")
    st.write(f"**Predicted Sentiment:** {selected_row['predicted_sentiment']} (Confidence: {selected_row['predicted_confidence']:.2f})")
    st.write(f"**Trust Score:** {selected_row['trust_score']:.2f}")
    st.write(f"**Explanation:** {selected_row['trust_explanation']}")
    st.write("**Full Details:**")
    st.json(selected_row.drop('trust_explanation').to_dict())
