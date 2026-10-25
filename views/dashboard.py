from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

import config
from views._ui import chip, go, metric_card, page_header, section

NAVY, GREEN, GREY = "#1F4E8C", "#1E8E5A", "#C8D0DC"


def render(svc) -> None:
    page_header(config.APP_TITLE, config.APP_SUBTITLE)
    today = date.today().strftime("%Y-%m-%d")
    total = svc.reports.total_registered()
    s = svc.reports.daily_summary(today)

    a, b = st.columns(2)
    a.button("▶ Launch live attendance", on_click=go, args=("Live Attendance",), width="stretch", type="primary")
    b.button("＋ Register a student", on_click=go, args=("Register Student",), width="stretch")

    cols = st.columns(4)
    cards = [("Registered students", total, "all records"),
             ("Present today", s["present"], today),
             ("Absent today", s["absent"], f"of {s['enrolled']} enrolled"),
             ("Attendance today", f"{s['percent']}%", "present ÷ enrolled")]
    for col, (l, v, h) in zip(cols, cards):
        col.markdown(metric_card(l, v, h), unsafe_allow_html=True)

    if total == 0:
        st.info("No students are registered yet. Use **Register Student** to add the first one.")

    left, right = st.columns([3, 2])
    with left:
        section("Attendance over the last 7 days")
        week = svc.reports.last_days(7)
        long = week.melt("date", ["present", "absent"], var_name="status", value_name="students")
        chart = (alt.Chart(long).mark_bar().encode(
            x=alt.X("date:N", title=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("students:Q", title="Students", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color("status:N", scale=alt.Scale(domain=["present", "absent"], range=[GREEN, GREY]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["date", "status", "students"]).properties(height=280))
        st.altair_chart(chart, width="stretch")
    with right:
        section("Today: present vs absent")
        if s["enrolled"] == 0:
            st.caption("No enrolled students to chart yet.")
        else:
            df = pd.DataFrame({"status": ["Present", "Absent"], "students": [s["present"], s["absent"]]})
            donut = (alt.Chart(df).mark_arc(innerRadius=60).encode(
                theta="students:Q",
                color=alt.Color("status:N", scale=alt.Scale(domain=["Present", "Absent"], range=[GREEN, GREY]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["status", "students"]).properties(height=280))
            st.altair_chart(donut, width="stretch")

    section("Recent attendance activity")
    recent = svc.attendance.recent(10)
    if recent.empty:
        st.caption("No attendance has been recorded yet.")
    else:
        st.dataframe(recent, hide_index=True, width="stretch")
