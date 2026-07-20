from __future__ import annotations

from typing import Any, Dict, Optional

from data.recovery_engine import assess_disruption_impact, find_recovery_options, get_disruption_cascade
from data.flights_db import get_flight


def answer_recovery_query(question: str, flight_ids: Optional[list] = None) -> str:
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

            impact = assess_disruption_impact(flight_id=fid.upper())
            if impact and impact.get("loaded"):
                lines.append(
                    f"**{fid.upper()}** disruption impact:\n"
                    f"  Route: {impact.get('origin', '?')} -> {impact.get('destination', '?')}\n"
                    f"  Delay: {impact.get('delay_min', 0)} min ({impact.get('disruption_reason', 'Unknown')})\n"
                    f"  Passengers affected: {impact.get('passengers_affected', 0)}\n"
                    f"  Connecting at risk: {impact.get('connecting_passengers_at_risk', 0)}\n"
                    f"  Baggage to reroute: {impact.get('baggage_needs_reroute', 0)}\n"
                    f"  Ground staff assigned: {impact.get('ground_staff_assigned', 0)}\n"
                    f"  Maintenance issues: {impact.get('maintenance_issues', 0)}\n"
                    f"  Airworthy: {'Yes' if impact.get('airworthy', True) else 'No'}\n"
                    f"  Retail revenue at risk: Rs {impact.get('retail_revenue_impact', 0):.0f}"
                )
            else:
                lines.append(f"No disruption impact data available for **{fid.upper()}**.")

            cascade = get_disruption_cascade(flight_id=fid.upper())
            if cascade and cascade.get("loaded"):
                severity = cascade.get("cascade_severity", "Low")
                same_ac = cascade.get("affected_flights_same_aircraft", [])
                staff_flights = cascade.get("staff_other_flights", [])
                lines.append(
                    f"  **Cascade** (severity: {severity}):\n"
                    f"    Same aircraft flights: {len(same_ac)}\n"
                    f"    Staff affected: {len(cascade.get('affected_staff', []))}\n"
                    f"    Staff other flights: {len(staff_flights)}"
                )
                if same_ac:
                    ac_flights = ", ".join(f["flight_id"] for f in same_ac[:5])
                    lines.append(f"    Same aircraft: {ac_flights}")
        return "\n\n".join(lines)

    if "recovery" in lowered or "option" in lowered or "plan" in lowered:
        if flight_ids:
            result = find_recovery_options(flight_id=flight_ids[0].upper())
            if not result or not result.get("loaded"):
                return f"No recovery options found for {flight_ids[0].upper()}."
            recs = result.get("recommendations", [])
            lines = [f"  **{r['type']}**: {r['message']}" for r in recs]
            return f"**Recovery options for {flight_ids[0].upper()}** ({result['available_staff_count']} staff available, {result['loaded_bags_to_reroute']} bags to reroute, {result['maintenance_issues']} mtc issues):\n" + "\n".join(lines)

        return "Please specify a flight ID to find recovery options for."

    if "cascade" in lowered or "impact" in lowered:
        if flight_ids:
            cascade = get_disruption_cascade(flight_id=flight_ids[0].upper())
            if not cascade or not cascade.get("loaded"):
                return f"No cascade effect for {flight_ids[0].upper()}."
            same_ac = cascade.get("affected_flights_same_aircraft", [])
            staff_flights = cascade.get("staff_other_flights", [])
            lines = [f"  **{f['flight_id']}**: {f['origin']}->{f['destination']} at {f.get('std', '?')}" for f in same_ac]
            staff_lines = [f"  Staff {s['staff_id']} -> flight {s['flight_id']}" for s in staff_flights[:5]]
            return (
                f"**Cascade for {flight_ids[0].upper()}** (severity: {cascade.get('cascade_severity', 'Low')}):\n"
                f"Aircraft: {cascade.get('aircraft_reg', 'N/A')}\n"
                f"Passengers affected: {cascade.get('passengers_affected', 0)}\n\n"
                f"**Same aircraft flights** ({len(same_ac)}):\n" + ("\n".join(lines) if lines else "  None") +
                f"\n\n**Staff other flights** ({len(staff_flights)}):\n" + ("\n".join(staff_lines) if staff_lines else "  None")
            )

        return "Please specify a flight ID to analyze cascade effects."

    return "I can help with disruption impact assessment, cascade analysis, or recovery planning. Please provide a flight ID to analyze."
