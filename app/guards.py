"""Calling-hours and do-not-call guards.

Hours follow the ACMA telemarketing standard for the callee's local time:
weekdays 09:00–20:00, Saturday 09:00–17:00, never Sunday, never a national
public holiday. The end of each window is exclusive.

National holidays come from the ``holidays`` calendar for Australia (the
days common to the country, not state holidays such as the King's Birthday).
When New Year's Day, Australia Day, Christmas Day, or Boxing Day falls on a
weekend, the following weekday is also blocked — the usual Australian
substitute day. If that weekday is already a holiday, the substitute moves
to the next free weekday.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import holidays

_SUBSTITUTED = frozenset(
    {"New Year's Day", "Australia Day", "Christmas Day", "Boxing Day"}
)


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    code: str
    message: str


def utcnow() -> datetime:
    return datetime.now(ZoneInfo("UTC"))


class NationalHolidayCalendar:
    """Australian national public holidays, plus weekend substitute days."""

    def __init__(self) -> None:
        self._years: dict[int, dict[date, str]] = {}

    def name(self, day: date) -> str | None:
        return self._year(day.year).get(day)

    def _year(self, year: int) -> dict[date, str]:
        cached = self._years.get(year)
        if cached is not None:
            return cached
        official = holidays.country_holidays("AU", years=[year])
        blocked: dict[date, str] = {day: str(name) for day, name in official.items()}
        for day, name in list(blocked.items()):
            base = name.split(" (")[0]
            if base not in _SUBSTITUTED or "observed" in name.lower():
                continue
            if day.weekday() == 5:
                observed = day + timedelta(days=2)
            elif day.weekday() == 6:
                observed = day + timedelta(days=1)
            else:
                continue
            while observed in blocked:
                observed += timedelta(days=1)
            blocked[observed] = f"{base} (observed)"
        self._years[year] = blocked
        return blocked


_CALENDAR = NationalHolidayCalendar()


def calling_window(
    moment: datetime,
    timezone_name: str,
    calendar: NationalHolidayCalendar | None = None,
) -> GuardResult:
    """Decide whether ``moment`` is inside the callee's calling window."""
    if moment.tzinfo is None:
        raise ValueError("moment must be timezone-aware")
    zone = ZoneInfo(timezone_name)
    local = moment.astimezone(zone)
    calendar = calendar or _CALENDAR
    holiday = calendar.name(local.date())
    if holiday:
        return GuardResult(
            False,
            "outside_calling_hours",
            f"National public holiday ({holiday}) in {timezone_name} "
            f"on {local.strftime('%A %Y-%m-%d')}.",
        )
    weekday = local.weekday()
    if weekday == 6:
        return GuardResult(
            False,
            "outside_calling_hours",
            f"Sunday in {timezone_name} ({local.strftime('%Y-%m-%d %H:%M')}).",
        )
    minutes = local.hour * 60 + local.minute
    if weekday == 5:
        if 9 * 60 <= minutes < 17 * 60:
            return GuardResult(True, "ok", "Inside Saturday calling hours.")
        return GuardResult(
            False,
            "outside_calling_hours",
            f"Outside Saturday hours (09:00–17:00) in {timezone_name} "
            f"({local.strftime('%Y-%m-%d %H:%M')}).",
        )
    if 9 * 60 <= minutes < 20 * 60:
        return GuardResult(True, "ok", "Inside weekday calling hours.")
    return GuardResult(
        False,
        "outside_calling_hours",
        f"Outside weekday hours (09:00–20:00) in {timezone_name} "
        f"({local.strftime('%A %Y-%m-%d %H:%M')}).",
    )


def evaluate_dial(
    *,
    number: str,
    timezone_name: str,
    moment: datetime,
    blocked_numbers: set[str],
    from_number: str,
    calendar: NationalHolidayCalendar | None = None,
) -> GuardResult:
    """Combine the do-not-call list and the calling window."""
    if number == from_number:
        return GuardResult(
            False,
            "invalid_number",
            "Refusing to dial the caller ID number.",
        )
    if number in blocked_numbers:
        return GuardResult(
            False,
            "do_not_call",
            f"{number} is on the do-not-call list.",
        )
    return calling_window(moment, timezone_name, calendar)
