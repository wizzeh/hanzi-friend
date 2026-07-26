"""Shape our review history into the numbers and chart geometry the
stats page renders. All charts are plain SVG; we compute coordinates
here and the template just emits shapes."""

import time
from datetime import date, timedelta

import db as store
import grammar

HEATMAP_WEEKS = 52
FORECAST_DAYS = 14
ACTIVITY_DAYS = 30
PASS_RATE_DAYS = 30

# Chart canvas; viewBox units, scaled by the page.
CHART_W = 620
CHART_H = 150
PAD_LEFT = 36
PAD_RIGHT = 14
PAD_BOTTOM = 18


def _heatmap_level(count: int) -> int:
    if count == 0:
        return 0
    if count < 10:
        return 1
    if count < 20:
        return 2
    if count < 40:
        return 3
    return 4


def heatmap(day_counts, today: date):
    """A year of review counts as week-columns of 7 day-cells."""
    start = today - timedelta(days=today.weekday(), weeks=HEATMAP_WEEKS - 1)
    columns = []
    month_labels = []
    for week in range(HEATMAP_WEEKS):
        column = []
        label = ""
        for weekday in range(7):
            day = start + timedelta(weeks=week, days=weekday)
            if day.day == 1:
                label = day.strftime("%b")
            if day > today:
                column.append(None)
                continue
            count = day_counts.get(day.isoformat(), 0)
            column.append(
                {
                    "date": day.isoformat(),
                    "count": count,
                    "level": _heatmap_level(count),
                }
            )
        columns.append(column)
        month_labels.append(label)
    return {"columns": columns, "month_labels": month_labels}


def streaks(review_day_set, today: date):
    """Current run of review days (alive if we reviewed today or
    yesterday) and the best run ever."""
    current = 0
    day = today
    if day.isoformat() not in review_day_set:
        day -= timedelta(days=1)
    while day.isoformat() in review_day_set:
        current += 1
        day -= timedelta(days=1)

    best = run = 0
    previous = None
    for iso in sorted(review_day_set):
        day = date.fromisoformat(iso)
        run = run + 1 if previous == day - timedelta(days=1) else 1
        best = max(best, run)
        previous = day
    return current, best


def cumulative_chart(learned_rows, today: date):
    """Area chart of total words known over time, with axis ticks."""
    if not learned_rows:
        return None

    first = date.fromisoformat(learned_rows[0]["day"])
    span = max((today - first).days, 1)
    total = sum(row["words"] for row in learned_rows)

    inner_w = CHART_W - PAD_LEFT - PAD_RIGHT
    inner_h = CHART_H - PAD_BOTTOM
    x = lambda day: PAD_LEFT + (day - first).days / span * inner_w
    y = lambda count: inner_h - count / total * (inner_h - 10)

    running = 0
    points = []
    for row in learned_rows:
        running += row["words"]
        points.append((x(date.fromisoformat(row["day"])), y(running)))
    points.append((x(today), y(running)))

    line = "M" + " L".join("{:.1f},{:.1f}".format(px, py) for px, py in points)
    area = (
        line
        + " L{:.1f},{} L{:.1f},{} Z".format(points[-1][0], inner_h, points[0][0], inner_h)
    )

    # A y gridline at a round step near a quarter of the range.
    step = max(round(total / 4, -len(str(total // 4)) + 1), 1) if total >= 4 else 1
    y_ticks = []
    count = step
    while count <= total:
        y_ticks.append({"value": int(count), "y": y(count)})
        count += step

    x_ticks = []
    month = date(first.year, first.month, 1)
    while month <= today:
        if month >= first and (span < 240 or month.month % 3 == 1):
            x_ticks.append({"label": month.strftime("%b %y"), "x": x(month)})
        month = (month + timedelta(days=32)).replace(day=1)

    return {
        "line": line,
        "area": area,
        "total": total,
        "y_ticks": y_ticks,
        "x_ticks": x_ticks,
        "baseline": inner_h,
    }


def bar_chart(rows, today: date, days: int, key: str, second_key=None):
    """Last-N-days bars; each bar carries its height and an optional
    inner bar (lapses) in viewBox units."""
    counts = {row["day"]: row for row in rows}
    peak = max([row[key] for row in rows] + [1])

    inner_w = CHART_W - PAD_LEFT - PAD_RIGHT
    inner_h = CHART_H - PAD_BOTTOM
    slot = inner_w / days
    bars = []
    for i in range(days):
        day = today - timedelta(days=days - 1 - i)
        row = counts.get(day.isoformat())
        value = row[key] if row else 0
        second = row[second_key] if row and second_key else 0
        height = value / peak * (inner_h - 10)
        bars.append(
            {
                "date": day.isoformat(),
                "label": day.strftime("%-d") if day.day in (1, 8, 15, 22) or i in (0, days - 1) else "",
                "value": value,
                "second": second,
                "x": PAD_LEFT + i * slot + slot * 0.15,
                "w": slot * 0.7,
                "y": inner_h - height,
                "h": height,
                "second_h": (second / peak * (inner_h - 10)) if second else 0,
            }
        )
    return {"bars": bars, "peak": peak, "baseline": inner_h}


def forecast_chart(user_id: int, today: date):
    """Overdue backlog plus cards coming due over the next two weeks."""
    horizon = time.mktime(
        (today + timedelta(days=FORECAST_DAYS)).timetuple()
    )
    rows = store.due_counts_by_day(user_id, horizon)

    overdue = 0
    per_day = {}
    for row in rows:
        if date.fromisoformat(row["day"]) <= today:
            overdue += row["cards"]
        else:
            per_day[row["day"]] = row["cards"]

    peak = max(list(per_day.values()) + [overdue, 1])
    inner_w = CHART_W - PAD_LEFT - PAD_RIGHT
    inner_h = CHART_H - PAD_BOTTOM
    slot = inner_w / (FORECAST_DAYS + 1)

    bars = []
    for i in range(FORECAST_DAYS + 1):
        day = today + timedelta(days=i)
        value = overdue if i == 0 else per_day.get(day.isoformat(), 0)
        height = value / peak * (inner_h - 10)
        bars.append(
            {
                "label": "now" if i == 0 else day.strftime("%-d"),
                "value": value,
                "overdue": i == 0,
                "x": PAD_LEFT + i * slot + slot * 0.15,
                "w": slot * 0.7,
                "y": inner_h - height,
                "h": height,
            }
        )
    return {"bars": bars, "peak": peak, "baseline": inner_h, "overdue": overdue}


def page_data(user_id: int, describe):
    """Everything stats.html needs. `describe` turns a word into
    {char, reading, gloss} for the hardest-words list."""
    today = date.today()
    days = store.review_days(user_id)
    learned = store.words_learned_by_day(user_id)

    day_counts = {row["day"]: row["reviews"] for row in days}
    total_reviews = sum(row["reviews"] for row in days)

    window = today - timedelta(days=PASS_RATE_DAYS)
    recent = [row for row in days if date.fromisoformat(row["day"]) > window]
    recent_total = sum(row["reviews"] for row in recent)
    recent_lapses = sum(row["lapses"] for row in recent)
    # One decimal so a rare lapse doesn't read as a perfect 100%.
    pass_rate = (
        "%g" % round(100 * (1 - recent_lapses / recent_total), 1)
        if recent_total
        else None
    )

    current_streak, best_streak = streaks(set(day_counts), today)

    hardest = []
    seen = set()
    for word in store.hardest_words(user_id, "translation-chinese"):
        if word in seen:
            continue
        seen.add(word)
        hardest.append(describe(word))
        if len(hardest) == 10:
            break

    grammar_seen, _ = store.grammar_state(user_id)
    grammar_backlog = len(
        grammar.backlog(grammar_seen, store.characters_seen(user_id))
    )

    return {
        "words_known": len(store.user_words(user_id)),
        "grammar_seen": grammar_seen,
        "grammar_backlog": grammar_backlog,
        "total_reviews": total_reviews,
        "review_days": len(days),
        "pass_rate": pass_rate,
        "current_streak": current_streak,
        "best_streak": best_streak,
        "heatmap": heatmap(day_counts, today),
        "cumulative": cumulative_chart(learned, today),
        "activity": bar_chart(days, today, ACTIVITY_DAYS, "reviews", "lapses"),
        "forecast": forecast_chart(user_id, today),
        "hardest": hardest,
        "chart_w": CHART_W,
        "chart_h": CHART_H,
        "pad_left": PAD_LEFT,
    }
