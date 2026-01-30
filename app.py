import itertools
import math
import re
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from googleapiclient.discovery import build


STOPWORDS = {
    "a","about","after","all","also","an","and","are","as","at","be","because","been",
    "but","by","can","could","for","from","get","how","if","in","into","is","it","its",
    "just","like","make","new","not","of","on","or","our","out","over","the","this",
    "to","today","top","up","what","when","why","with","you","your",
}

TOPIC_TEMPLATES = [
    "Beginner's guide to {kw1}",
    "{kw1} vs {kw2}: which should you choose?",
    "How to master {kw1} in 30 days",
    "The truth about {kw1} and {kw2}",
    "{kw1} mistakes creators keep making",
    "{kw1} trends to watch this year",
    "Step-by-step {kw1} workflow",
    "{kw1} tools that save hours",
]


# ---------- UI SAFE RENDERERS (NO ARROW) ----------

def render_as_markdown_table(headers: List[str], rows: List[List[str]]):
    """Render table as Markdown to avoid Arrow serialization issues."""
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    st.markdown("\n".join([header_line, sep_line] + body_lines))


def safe_text(x) -> str:
    if x is None:
        return ""
    return str(x)


def fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "0"


def fmt_float(x, ndigits=3) -> str:
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return "0.000"


def video_link_from_title(title: str, video_id: str) -> str:
    # Markdown link
    return f"[{safe_text(title)}](https://www.youtube.com/watch?v={safe_text(video_id)})"


# ---------- YOUTUBE HELPERS ----------

def build_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


def extract_video_id(video_url: str) -> str:
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, video_url)
        if m:
            return m.group(1)
    raise ValueError("Invalid YouTube video URL. Example: https://youtu.be/VIDEOID")


def get_channel_id_from_video(client, video_url: str) -> str:
    video_id = extract_video_id(video_url)
    resp = client.videos().list(part="snippet", id=video_id).execute()
    items = resp.get("items", [])
    if not items:
        raise ValueError("No video found for this URL.")
    return items[0]["snippet"]["channelId"]


def get_uploads_playlist_id(client, channel_id: str) -> str:
    resp = client.channels().list(part="contentDetails", id=channel_id).execute()
    items = resp.get("items", [])
    if not items:
        raise ValueError("Channel not found.")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def fetch_playlist_video_ids(client, playlist_id: str, max_videos: int) -> List[str]:
    ids = []
    token = None
    while len(ids) < max_videos:
        resp = client.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=min(50, max_videos - len(ids)),
            pageToken=token,
        ).execute()

        for item in resp.get("items", []):
            ids.append(item["contentDetails"]["videoId"])

        token = resp.get("nextPageToken")
        if not token:
            break
    return ids


def fetch_video_details(client, video_ids: List[str]) -> List[Dict]:
    details = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        resp = client.videos().list(part="snippet,statistics", id=",".join(chunk)).execute()
        details.extend(resp.get("items", []))
    return details


# ---------- DATA PROCESSING ----------

def build_dataframe(details: List[Dict]) -> pd.DataFrame:
    rows = []
    for item in details:
        snip = item.get("snippet", {})
        stats = item.get("statistics", {})
        rows.append({
            "video_id": item.get("id"),
            "title": snip.get("title", ""),
            "published_at": snip.get("publishedAt"),
            "tags": ", ".join(snip.get("tags", [])) if snip.get("tags") else "",
            "views": int(stats.get("viewCount", 0)),
            "likes": int(stats.get("likeCount", 0)),
            "comments": int(stats.get("commentCount", 0)),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df = df.dropna(subset=["published_at"])
        df.sort_values("published_at", inplace=True)
    return df


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return dict(avg_views=0, avg_likes=0, avg_comments=0, like_comment_ratio=0)

    avg_views = float(df["views"].mean())
    avg_likes = float(df["likes"].mean())
    avg_comments = float(df["comments"].mean())
    ratio = avg_likes / avg_comments if avg_comments else math.nan
    return {
        "avg_views": avg_views,
        "avg_likes": avg_likes,
        "avg_comments": avg_comments,
        "like_comment_ratio": ratio,
    }


def add_engagement_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Engagement rate per video = (likes + comments) / views."""
    if df.empty:
        return df
    df2 = df.copy()
    df2["engagement_rate"] = (df2["likes"] + df2["comments"]) / df2["views"].replace(0, pd.NA)
    df2["engagement_rate"] = df2["engagement_rate"].fillna(0.0).astype(float)
    return df2


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]+", safe_text(text).lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def weighted_keyword_scores(df: pd.DataFrame) -> Counter:
    now = pd.Timestamp.utcnow()
    scores = Counter()

    for _, row in df.iterrows():
        published = row["published_at"]
        if pd.isna(published):
            continue

        months_since = (now - published).days / 30
        weight = 1 / (1 + months_since)

        tokens = tokenize(row["title"]) + tokenize(row["tags"])
        for t in tokens:
            scores[t] += weight

    return scores


def build_topic_predictions(df: pd.DataFrame, max_topics: int = 8) -> Tuple[List[str], List[Tuple[str, float]]]:
    if df.empty:
        return [], []

    scores = weighted_keyword_scores(df)
    top_keywords = scores.most_common(8)
    keywords = [kw for kw, _ in top_keywords]

    if len(keywords) < 2:
        return [], top_keywords

    topics = []
    for template, pair in zip(TOPIC_TEMPLATES, itertools.cycle(itertools.combinations(keywords, 2))):
        kw1, kw2 = pair
        topic = template.format(kw1=kw1.title(), kw2=kw2.title())
        if topic not in topics:
            topics.append(topic)
        if len(topics) >= max_topics:
            break

    return topics, top_keywords


# ---------- PLOTS ----------

def plot_upload_frequency(df: pd.DataFrame):
    if df.empty:
        st.info("No videos to plot.")
        return

    df_monthly = (
        df.set_index("published_at")
          .resample("ME")      # Pandas 3.x month-end supported
          .size()
          .rename("videos")
          .reset_index()
    )
    st.line_chart(df_monthly, x="published_at", y="videos")


def plot_views_over_time(df: pd.DataFrame):
    if df.empty:
        st.info("No videos to plot.")
        return
    st.line_chart(df, x="published_at", y="views")


def plot_best_day_hour(df: pd.DataFrame):
    """Best upload day/time analysis."""
    if df.empty:
        st.info("No videos to analyze.")
        return

    df2 = df.copy()
    df2["day_name"] = df2["published_at"].dt.day_name()
    df2["hour"] = df2["published_at"].dt.hour

    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_views = df2.groupby("day_name")["views"].mean().reindex(day_order).fillna(0)

    hour_views = df2.groupby("hour")["views"].mean().reindex(range(24)).fillna(0)

    colA, colB = st.columns(2)
    with colA:
        st.write("**Avg Views by Upload Day**")
        st.bar_chart(day_views)
        best_day = day_views.idxmax() if day_views.sum() > 0 else "N/A"
        st.caption(f"Best day (highest avg views): **{best_day}**")

    with colB:
        st.write("**Avg Views by Upload Hour (UTC)**")
        st.bar_chart(hour_views)
        best_hour = int(hour_views.idxmax()) if hour_views.sum() > 0 else None
        st.caption(f"Best hour (UTC): **{best_hour}:00**" if best_hour is not None else "Best hour: N/A")


# ---------- NEW FEATURES: BUCKETS + TOP LISTS + DOWNLOAD ----------

def assign_performance_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buckets based on views vs channel avg:
      Viral: >= 2x avg
      Good:  >= 1.25x avg and < 2x avg
      Average: >= 0.75x avg and < 1.25x avg
      Underperforming: < 0.75x avg
    """
    if df.empty:
        return df
    df2 = df.copy()
    avg = float(df2["views"].mean()) if len(df2) else 0.0

    def bucket(v):
        if avg <= 0:
            return "Average"
        if v >= 2.0 * avg:
            return "Viral"
        if v >= 1.25 * avg:
            return "Good"
        if v >= 0.75 * avg:
            return "Average"
        return "Underperforming"

    df2["bucket"] = df2["views"].apply(bucket)
    return df2


def show_bucket_summary(df: pd.DataFrame):
    if df.empty or "bucket" not in df.columns:
        st.info("No bucket data available.")
        return

    counts = df["bucket"].value_counts().to_dict()
    rows = [[k, str(counts.get(k, 0))] for k in ["Viral","Good","Average","Underperforming"]]
    render_as_markdown_table(["Performance Bucket", "Video Count"], rows)


def top5_tables(df: pd.DataFrame):
    if df.empty:
        st.info("No videos to rank.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Top 5 by Views**")
        top = df.sort_values("views", ascending=False).head(5)
        rows = [[video_link_from_title(r["title"], r["video_id"]), fmt_int(r["views"])] for _, r in top.iterrows()]
        render_as_markdown_table(["Video", "Views"], rows)

    with col2:
        st.write("**Top 5 by Likes**")
        top = df.sort_values("likes", ascending=False).head(5)
        rows = [[video_link_from_title(r["title"], r["video_id"]), fmt_int(r["likes"])] for _, r in top.iterrows()]
        render_as_markdown_table(["Video", "Likes"], rows)

    with col3:
        st.write("**Top 5 by Comments**")
        top = df.sort_values("comments", ascending=False).head(5)
        rows = [[video_link_from_title(r["title"], r["video_id"]), fmt_int(r["comments"])] for _, r in top.iterrows()]
        render_as_markdown_table(["Video", "Comments"], rows)


def show_top_engaging(df: pd.DataFrame):
    """Engagement Rate + Top Engaging Videos"""
    if df.empty or "engagement_rate" not in df.columns:
        st.info("No engagement data available.")
        return

    avg_eng = float(df["engagement_rate"].mean()) if len(df) else 0.0
    st.metric("Average Engagement Rate", f"{avg_eng*100:.2f}%")

    top = df.sort_values("engagement_rate", ascending=False).head(10)
    rows = []
    for _, r in top.iterrows():
        rows.append([
            video_link_from_title(r["title"], r["video_id"]),
            f"{r['engagement_rate']*100:.2f}%",
            fmt_int(r["views"]),
            fmt_int(r["likes"]),
            fmt_int(r["comments"]),
        ])
    render_as_markdown_table(["Video", "Engagement", "Views", "Likes", "Comments"], rows)


def download_csv_buttons(df: pd.DataFrame):
    if df.empty:
        return

    # Safe: convert datetime to string to avoid Arrow issues during conversion/rendering
    export_df = df.copy()
    if "published_at" in export_df.columns:
        export_df["published_at"] = export_df["published_at"].astype(str)

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Full Data as CSV",
        data=csv_bytes,
        file_name="youtube_insights.csv",
        mime="text/csv",
    )


# ---------- STREAMLIT APP ----------

def main():
    st.set_page_config(page_title="YouTube Insight Generator", layout="wide")
    st.title("📊 YouTube Video Analysis & Insight Generator")

    with st.sidebar:
        st.header("Inputs")
        api_key = st.text_input("YouTube Data API Key", type="password")
        video_url = st.text_input("YouTube Video URL")
        max_videos = st.slider("Number of recent videos", 10, 100, 50)
        fetch = st.button("Fetch Insights")

    if not fetch:
        st.info("Enter API key + any YouTube video URL, then click **Fetch Insights**.")
        return

    if not api_key or not video_url:
        st.error("Please provide both API key and video URL.")
        return

    with st.spinner("Fetching data from YouTube..."):
        try:
            client = build_client(api_key)
            channel_id = get_channel_id_from_video(client, video_url)
            playlist_id = get_uploads_playlist_id(client, channel_id)
            video_ids = fetch_playlist_video_ids(client, playlist_id, max_videos)
            if not video_ids:
                st.error("No videos found for this channel.")
                return
            details = fetch_video_details(client, video_ids)
            df = build_dataframe(details)
        except Exception as e:
            st.error("Failed to fetch YouTube data. Check API key / quota / URL.")
            st.exception(e)
            return

    # Keep original order/features exactly:
    # 1) channel summary
    st.subheader("Channel Summary")
    metrics = compute_metrics(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Views", f"{metrics['avg_views']:.0f}")
    c2.metric("Avg Likes", f"{metrics['avg_likes']:.0f}")
    c3.metric("Avg Comments", f"{metrics['avg_comments']:.0f}")
    ratio = metrics["like_comment_ratio"]
    c4.metric("Like/Comment Ratio", f"{ratio:.2f}" if math.isfinite(ratio) else "N/A")

    # NEW: Download CSV (placed right after summary; old features unaffected)
    st.subheader("Download")
    download_csv_buttons(df)

    # 2) upload frequency plot
    st.subheader("📅 Upload Frequency")
    plot_upload_frequency(df)

    # NEW: Best upload day/time (doesn't touch old plots)
    st.subheader("🕒 Best Upload Day / Time")
    plot_best_day_hour(df)

    # 3) views over time plot
    st.subheader("📈 Views Over Time")
    plot_views_over_time(df)

    # NEW: Engagement rate + top engaging
    st.subheader("🔥 Engagement Rate + Top Engaging Videos")
    df_eng = add_engagement_rate(df)
    show_top_engaging(df_eng)

    # NEW: Performance buckets
    st.subheader("🎯 Performance Buckets")
    df_bucket = assign_performance_buckets(df)
    show_bucket_summary(df_bucket)

    # NEW: Top 5 videos (views/likes/comments)
    st.subheader("🏆 Top 5 Videos")
    top5_tables(df)

    # 4) next video topic predictions (original)
    st.subheader("💡 Next Video Topic Predictions")
    topics, keywords = build_topic_predictions(df)

    if topics:
        st.write("Suggested topics:")
        for t in topics:
            st.markdown(f"- **{t}**")
    else:
        st.info("Not enough keywords to generate topic suggestions.")

    # 5) top weighted keywords (original style)
    st.subheader("Top weighted keywords")
    kw_rows = [[safe_text(k), f"{float(s):.3f}"] for k, s in keywords]
    if kw_rows:
        render_as_markdown_table(["Keyword", "Score"], kw_rows)
    else:
        st.info("No keywords available.")

    # 6) latest videos list (original style)
    st.subheader("📄 Latest Videos")
    if df.empty:
        st.info("No videos to show.")
    else:
        df_latest = df.sort_values("published_at", ascending=False).head(12).copy()
        for _, r in df_latest.iterrows():
            st.markdown(
                f"**{safe_text(r['title'])}**  \n"
                f"- Published: {safe_text(r['published_at'])}  \n"
                f"- Views: {int(r['views'])} | Likes: {int(r['likes'])} | Comments: {int(r['comments'])}"
            )
            st.markdown("---")


if __name__ == "__main__":
    main()
