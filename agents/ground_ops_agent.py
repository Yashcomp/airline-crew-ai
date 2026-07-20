from __future__ import annotations

from typing import Any, Dict, Optional

from data.ground_ops.turnaround_analytics import get_turnaround_profile, get_boarding_efficiency, get_maintenance_impact
from data.ground_ops.staff_analytics import get_staff_role_distribution, get_shift_coverage_analysis, identify_understaffed_periods
from data.flights_db import get_flight


def answer_ground_ops_query(question: str, flight_ids: Optional[list] = None) -> str:
    lowered = question.lower()

    if flight_ids:
        lines = []
        for fid in flight_ids:
            flight = get_flight(fid.upper().replace("_", "-"))
            if not flight:
                flight = get_flight(fid.upper())
            if not flight:
                lines.append(f"Flight **{fid.upper()}** not found in schedule.")
                continue

            profile = get_turnaround_profile(flight_id=fid.upper())
            if profile and not profile.get("error"):
                bag_sum = profile.get("baggage_summary", {})
                lines.append(
                    f"**{fid.upper()}** turnaround profile:\n"
                    f"  Route: {profile.get('origin', '?')} -> {profile.get('destination', '?')}\n"
                    f"  Boarding duration: {profile.get('boarding_duration_min', 0)} min\n"
                    f"  Baggage: {bag_sum.get('total_bags', 0)} bags, {bag_sum.get('total_weight_kg', 0)}kg\n"
                    f"  Maintenance issues: {profile.get('maintenance_issues', 0)}\n"
                    f"  Airworthy: {'Yes' if profile.get('airworthy', True) else 'No'}"
                )
            else:
                lines.append(f"No turnaround data available for **{fid.upper()}**.")
        return "\n\n".join(lines)

    if "boarding" in lowered or "efficiency" in lowered:
        result = get_boarding_efficiency()
        if not result or not result.get("loaded"):
            return "No boarding efficiency data available."
        flights = result.get("flights", [])
        if not flights:
            return f"No boarding events found. Avg boarding: {result.get('avg_boarding_min', 0)} min across {result.get('flights_analyzed', 0)} flights."
        lines = [f"**{r['flight_id']}**: {r['duration_min']} min boarding, {r['pax_count']}/{r['booked_pax']} pax ({r['aircraft_type']})" for r in flights]
        header = f"**Boarding efficiency** ({result['flights_analyzed']} flights, avg {result['avg_boarding_min']} min, {result['efficiency_per_pax']} min/pax):"
        return header + "\n" + "\n".join(lines)

    if "maintenance" in lowered or "impact" in lowered:
        result = get_maintenance_impact()
        if not result or not result.get("loaded"):
            return "No maintenance impact data available."
        aircraft = result.get("aircraft_maintenance", [])
        defects = result.get("top_defects", [])
        lines = [f"**{r['aircraft_reg']}**: {r['work_orders']} work orders, avg {r['avg_duration']:.1f}h, severity={r['avg_severity']:.1f}" for r in aircraft]
        defect_lines = [f"  {r['defect']} ({r['component']}): {r['cnt']}x, avg {r['avg_hrs']:.1f}h" for r in defects]
        return f"**Maintenance by aircraft** ({result['total_work_orders']} total orders):\n" + "\n".join(lines) + "\n\n**Top defects:**\n" + "\n".join(defect_lines)

    if "understaffed" in lowered or "shortage" in lowered:
        understaffed = identify_understaffed_periods()
        if not understaffed:
            return "No understaffed periods detected."
        lines = []
        for u in understaffed[:10]:
            lines.append(f"Hour {u.get('hour', '?'):02d}:00 — {u.get('staff', 0)} staff, {u.get('flights', 0)} flights (ratio: {u.get('staff_per_flight', 0)}, need {u.get('recommended', 0)})")
        return f"**Understaffed periods:**\n" + "\n".join(lines)

    if "staff" in lowered or "distribution" in lowered:
        result = get_staff_role_distribution()
        if not result or not result.get("loaded"):
            return "No staff data available."
        by_role = result.get("by_role", {})
        total = result.get("total_staff", 0)
        lines = [f"**{role}**: {count} ({count/total*100:.0f}%)" for role, count in by_role.items()]
        return f"**Staff distribution** ({total} total):\n" + "\n".join(lines)

    if "shift" in lowered or "coverage" in lowered:
        result = get_shift_coverage_analysis()
        if not result or not result.get("loaded"):
            return "No shift coverage data available."
        coverage = result.get("coverage_by_hour", [])
        lines = []
        for r in coverage[:12]:
            lines.append(f"Hour {r.get('hour', '?'):02d}:00 — {r.get('staff_on_duty', 0)} staff, {r.get('flights', 0)} flights, ratio={r.get('staff_per_flight', 0)}")
        return f"**Shift coverage (hourly):**\n" + "\n".join(lines)

    return "I can help with turnaround profiles, boarding efficiency, maintenance impact, staff distribution, shift coverage, or understaffing. What would you like to know?"
