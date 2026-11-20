import html
import base64
import plotly.graph_objects as go
import streamlit as st
from config import COLORS, LABELS, ROOT


def style():
    st.html((ROOT / "assets" / "style.css").read_text(encoding="utf-8"))


def heading(eyebrow, title, description):
    st.html(f'<div class="eyebrow">{html.escape(eyebrow)}</div><h1>{html.escape(title)}</h1>'
            f'<p class="subtitle">{html.escape(description)}</p>')


def card(label, value, detail, accent="cyan"):
    st.html(f'<div class="metric-card {accent}"><div class="metric-label">{html.escape(label)}</div>'
            f'<div class="metric-value">{html.escape(str(value))}</div><div class="metric-detail">{html.escape(detail)}</div></div>')


def chart_style(fig, height=300):
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Arial, sans-serif", size=14, color="#606060"), height=height,
                      margin=dict(l=12, r=12, t=20, b=15), showlegend=False)
    fig.update_xaxes(gridcolor="rgba(0,0,0,.07)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,.07)", zeroline=False)
    return fig


def probability_chart(result):
    fig = go.Figure(go.Bar(x=[result[f"{v.lower()}_score"] for v in LABELS], y=list(LABELS), orientation="h",
                          marker_color=[COLORS[v] for v in LABELS],
                          text=[f'{result[f"{v.lower()}_score"]:.1%}' for v in LABELS], textposition="auto"))
    chart_style(fig, 230)
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def sentiment_card(result):
    label = result["sentiment"]
    color = "#181818"
    st.html(f'<div class="prediction" style="--sentiment:{color}"><div class="eyebrow">MODEL PREDICTION</div>'
            f'<div class="prediction-label">{label}</div><div class="prediction-score">{result["confidence"]:.1%}'
            '<span> model confidence</span></div></div>')


def distribution(frame, prefix):
    counts = frame[prefix + "sentiment"].value_counts().reindex(LABELS, fill_value=0)
    a, b = st.columns([1, 1.3], gap="large")
    with a, st.container(border=True):
        st.markdown("#### Sentiment mix")
        st.caption("The balance of customer feedback")
        fig = go.Figure(go.Pie(labels=list(LABELS), values=counts.values, hole=.76, sort=False,
                               marker=dict(colors=[COLORS[v] for v in LABELS], line=dict(color="#f4f3f1", width=5)),
                               textinfo="none", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"))
        chart_style(fig)
        fig.add_annotation(text=f'<b>{int(counts.sum()):,}</b><br>reviews', x=.5, y=.5, showarrow=False, font_size=22)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown(" · ".join(f"{v}: **{counts[v]}**" for v in LABELS))
    with b, st.container(border=True):
        st.markdown("#### Feedback breakdown")
        st.caption("Review count across each sentiment")
        fig = go.Figure(go.Bar(x=list(LABELS), y=counts.values,
                              marker_color=[COLORS[v] for v in LABELS], width=.42,
                              text=counts.values, textposition="outside", cliponaxis=False))
        chart_style(fig)
        fig.update_yaxes(title="Reviews", dtick=max(1, int(counts.max()) // 4), rangemode="tozero")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Sentiment indicates tone, not a satisfaction rating.")


def sculpture_card():
    path = ROOT / "assets" / "sculpture.png"
    art = ('<img class="sculpture-art" alt="Black satin ribbon surrounding a stone sphere" src="data:image/png;base64,'
           + base64.b64encode(path.read_bytes()).decode() + '">') if path.exists() else '<div class="orbit"><b></b></div>'
    st.html('<div class="sculpture-card"><div class="sculpture-kicker">THE ART OF UNDERSTANDING <span>✧</span></div>'
            + art + '<div class="sculpture-caption"><strong>Signal from the noise.</strong><span>Three sentiments. A clearer perspective.</span></div></div>')
