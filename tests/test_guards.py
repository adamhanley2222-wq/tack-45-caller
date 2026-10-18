"""Calling-hours, holidays, phone normalisation, and the do-not-call check."""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.guards import NationalHolidayCalendar, calling_window, evaluate_dial, load_test_numbers
from app.phones import PhoneError, infer_timezone, normalize_phone

SYDNEY = "Australia/Sydney"


def at(year, month, day, hour, minute, zone=SYDNEY) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=ZoneInfo(zone))


def test_weekday_window_boundaries():
    calendar = NationalHolidayCalendar()
    # Thursday 15 Jan 2026.
    assert calling_window(at(2026, 1, 15, 8, 59), SYDNEY, calendar).allowed is False
    assert calling_window(at(2026, 1, 15, 9, 0), SYDNEY, calendar).allowed is True
    assert calling_window(at(2026, 1, 15, 19, 59), SYDNEY, calendar).allowed is True
    refused = calling_window(at(2026, 1, 15, 20, 0), SYDNEY, calendar)
    assert refused.allowed is False
    assert refused.code == "outside_calling_hours"


def test_saturday_window_and_sunday_block():
    calendar = NationalHolidayCalendar()
    # Saturday 17 Jan 2026, Sunday 18 Jan 2026.
    assert calling_window(at(2026, 1, 17, 9, 0), SYDNEY, calendar).allowed is True
    assert calling_window(at(2026, 1, 17, 16, 59), SYDNEY, calendar).allowed is True
    assert calling_window(at(2026, 1, 17, 17, 0), SYDNEY, calendar).allowed is False
    sunday = calling_window(at(2026, 1, 18, 10, 0), SYDNEY, calendar)
    assert sunday.allowed is False
    assert "Sunday" in sunday.message


def test_national_holidays_and_weekend_substitute():
    calendar = NationalHolidayCalendar()
    christmas = calling_window(at(2026, 12, 25, 10, 0), SYDNEY, calendar)
    assert christmas.allowed is False
    assert "Christmas Day" in christmas.message

    anzac = calling_window(at(2026, 4, 25, 10, 0), SYDNEY, calendar)
    assert anzac.allowed is False
    assert "ANZAC" in anzac.message

    good_friday = calling_window(at(2026, 4, 3, 11, 0), SYDNEY, calendar)
    assert good_friday.allowed is False

    # 1 Jan 2022 was a Saturday. The Monday substitute is also blocked.
    substitute = calling_window(at(2022, 1, 3, 10, 0), SYDNEY, calendar)
    assert substitute.allowed is False
    assert "observed" in substitute.message

    mourning = calling_window(at(2022, 9, 22, 10, 0), SYDNEY, calendar)
    assert mourning.allowed is False
    assert "Mourning" in mourning.message


def test_state_kings_birthday_is_not_a_national_block():
    calendar = NationalHolidayCalendar()
    # Second Monday of June 2026 is the NSW King's Birthday, not a national holiday.
    kings = calling_window(at(2026, 6, 8, 10, 0), SYDNEY, calendar)
    assert kings.allowed is True


def test_perth_clock_can_refuse_when_sydney_would_allow():
    calendar = NationalHolidayCalendar()
    # 09:30 in Sydney on a Thursday is 06:30 in Perth.
    moment = at(2026, 1, 15, 9, 30, "Australia/Sydney")
    assert calling_window(moment, "Australia/Sydney", calendar).allowed is True
    perth = calling_window(moment, "Australia/Perth", calendar)
    assert perth.allowed is False
    assert "Australia/Perth" in perth.message


def test_do_not_call_and_self_dial_beat_an_open_window():
    moment = at(2026, 1, 15, 10, 0)
    blocked = evaluate_dial(
        number="+61412345678",
        timezone_name=SYDNEY,
        moment=moment,
        blocked_numbers={"+61412345678"},
        from_number="+61290904464",
    )
    assert blocked.allowed is False
    assert blocked.code == "do_not_call"

    self_dial = evaluate_dial(
        number="+61290904464",
        timezone_name=SYDNEY,
        moment=moment,
        blocked_numbers=set(),
        from_number="+61290904464",
    )
    assert self_dial.code == "invalid_number"

    allowed = evaluate_dial(
        number="+61280000000",
        timezone_name=SYDNEY,
        moment=moment,
        blocked_numbers=set(),
        from_number="+61290904464",
    )
    assert allowed.allowed is True


def test_test_number_skips_sunday_but_not_do_not_call(caplog):
    sunday = at(2026, 1, 18, 11, 0)
    listed = {"+61425229662"}
    with caplog.at_level(logging.INFO, logger="tack45.guards"):
        allowed = evaluate_dial(
            number="+61425229662",
            timezone_name=SYDNEY,
            moment=sunday,
            blocked_numbers=set(),
            from_number="+61290904464",
            test_numbers=listed,
        )
    assert allowed.allowed is True
    assert "Calling-hours exemption for test number +61425229662" in caplog.text

    refused = evaluate_dial(
        number="+61280001111",
        timezone_name=SYDNEY,
        moment=sunday,
        blocked_numbers=set(),
        from_number="+61290904464",
        test_numbers=listed,
    )
    assert refused.allowed is False
    assert refused.code == "outside_calling_hours"

    blocked = evaluate_dial(
        number="+61425229662",
        timezone_name=SYDNEY,
        moment=sunday,
        blocked_numbers={"+61425229662"},
        from_number="+61290904464",
        test_numbers=listed,
    )
    assert blocked.allowed is False
    assert blocked.code == "do_not_call"


def test_test_numbers_file_and_env_are_combined(tmp_path):
    path = tmp_path / "test_numbers.txt"
    path.write_text(
        "# internal\n+61425229662  # Adam Hanley own mobile, internal testing only\n\n",
        encoding="utf-8",
    )
    numbers = load_test_numbers(str(path), "0412 000 222, not-a-number")
    assert numbers == {"+61425229662", "+61412000222"}


def test_normalize_australian_numbers():
    assert normalize_phone("+61 412 345 678") == "+61412345678"
    assert normalize_phone("0412345678") == "+61412345678"
    assert normalize_phone("(02) 9090 4464") == "+61290904464"
    assert normalize_phone("61412345678") == "+61412345678"
    assert normalize_phone("1300 123 456") == "+611300123456"
    with pytest.raises(PhoneError):
        normalize_phone("123")
    with pytest.raises(PhoneError):
        normalize_phone("")


def test_timezone_inference():
    assert infer_timezone("+61290904464", "Australia/Sydney")[0] == "Australia/Sydney"
    assert infer_timezone("+61399999999", "Australia/Sydney")[0] == "Australia/Melbourne"
    assert infer_timezone("+61733333333", "Australia/Sydney")[0] == "Australia/Brisbane"
    mobile_tz, mobile_warning = infer_timezone("+61412345678", "Australia/Sydney")
    assert mobile_tz == "Australia/Sydney"
    assert mobile_warning
    eight_tz, eight_warning = infer_timezone("+61899999999", "Australia/Perth")
    assert eight_tz == "Australia/Perth"
    assert "Perth" in eight_warning
    explicit = infer_timezone("+61412345678", "Australia/Adelaide")
    assert explicit[0] == "Australia/Adelaide"
