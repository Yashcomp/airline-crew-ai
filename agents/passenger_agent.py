from __future__ import annotations

from typing import Any, Dict, Optional

from data.ground_ops.flow_analytics import (
    get_passenger_profile, get_baggage_load_profile, get_demand_by_route,
    get_connecting_pax_analysis, predict_baggage_load,
)
from data.ground_ops.security_analytics import get_security_throughput, predict_queue_buildup
from data.ground_ops.retail_analytics import get_revenue_by_flight, get_passenger_spend_profile
from data.flights_db import get_flight


def answer_passenger_query(question: str, flight_ids: Optional[list] = None) -> str:
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

            pax = get_passenger_profile(flight_id=fid.upper())
            bag = get_baggage_load_profile(flight_id=fid.upper())
            pax_n = pax.get("total_passengers", 0) if pax and pax.get("loaded") else 0
            bag_n = bag.get("total_bags", 0) if bag and bag.get("loaded") else 0
            bag_w = bag.get("total_weight_kg", 0) if bag and bag.get("loaded") else 0
            lines.append(
                f"**{fid.upper()}** passenger/baggage profile:\n"
                f"  Passengers: {pax_n}\n"
                f"  Baggage items: {bag_n} ({bag_w}kg)\n"
                f"  Nationalities: {', '.join(list(pax.get('nationality_distribution', {}).keys())[:5]) if pax and pax.get('loaded') else 'N/A'}\n"
                f"  Connecting rate: {pax.get('connecting_rate', 0) * 100:.0f}%" if pax and pax.get("loaded") else ""
            )
        return "\n\n".join(lines)

    if "demand" in lowered or "route" in lowered:
        result = get_demand_by_route()
        if not result or not result.get("loaded"):
            return "No demand data available."
        routes = result.get("routes", {})
        lines = [f"**{route}**: {r['total_pax']} pax across {r['flights']} flights ({r['route_type']})" for route, r in list(routes.items())[:10]]
        return f"**Demand by route:**\n" + "\n".join(lines)

    if "connecting" in lowered or "connection" in lowered:
        result = get_connecting_pax_analysis()
        if not result or not result.get("loaded"):
            return "No connecting passenger data available."
        top_routes = result.get("top_connecting_routes", [])
        lines = [f"**{r['route']}**: {r['connecting']}/{r['total']} connecting ({r['rate']*100:.0f}%)" for r in top_routes[:10]]
        return f"**Connecting passengers** ({result['connecting_passengers']}/{result['total_passengers']}, {result['connecting_rate']*100:.0f}% rate):\n" + "\n".join(lines)

    if "baggage" in lowered or "bag" in lowered:
        if "predict" in lowered or "forecast" in lowered:
            result = predict_baggage_load()
            if not result or not result.get("loaded"):
                return "No baggage prediction data available."
            return f"**Baggage prediction**: {result['predicted_bags']} bags, {result['predicted_weight_kg']}kg predicted from {result['passengers']} pax ({result['avg_bags_per_pax']} bags/pax)"

        result = get_baggage_load_profile()
        if not result or not result.get("loaded"):
            return "No baggage data available."
        return f"**Baggage profile**: {result['total_bags']} bags, {result['total_weight_kg']}kg, avg {result['avg_weight_per_bag']}kg/bag, {result['delayed_bags']} delayed, {result['oversized_bags']} oversized"

    if "security" in lowered or "screening" in lowered:
        if "queue" in lowered:
            result = predict_queue_buildup()
            if not result or not result.get("loaded"):
                return "No queue prediction data available."
            predictions = result.get("predictions", [])
            lines = [f"Hour {r['hour']:02d}:00 — {r.get('departing_pax', 0)} departing pax, {r.get('est_queue_length', 0)} est queue" for r in predictions]
            return f"**Queue buildup prediction:**\n" + "\n".join(lines)

        result = get_security_throughput()
        if not result or not result.get("loaded"):
            return "No security throughput data available."
        hourly = result.get("hourly_throughput", [])
        lines = [f"Hour {r['hour']:02d}:00 — {r['passengers_screened']} screened, {r.get('avg_screen_time_min', 0):.1f} min avg" for r in hourly]
        return f"**Security throughput:**\n" + "\n".join(lines)

    if "revenue" in lowered or "retail" in lowered or "spend" in lowered:
        result = get_revenue_by_flight()
        if not result or not result.get("loaded"):
            return "No revenue data available."
        flights = result.get("by_flight", [])
        lines = [f"**{r['flight_id']}**: {r['txn_count']} txns, Rs {r['total_revenue']:.0f}" for r in flights[:10]]
        return f"**Retail revenue by flight:**\n" + "\n".join(lines)

    pax = get_passenger_profile()
    if pax and pax.get("loaded"):
        nat = pax.get("nationality_distribution", {})
        top_nats = sorted(nat.items(), key=lambda x: x[1], reverse=True)[:5]
        lines = [f"**{n}**: {c} passengers" for n, c in top_nats]
        return f"**Passenger profiles** ({pax['total_passengers']} total, {pax.get('avg_loyalty_score', 0)} avg loyalty):\n" + "\n".join(lines)

    return "I can help with passenger demand, baggage loads, connecting passengers, security throughput, queue predictions, revenue, or passenger profiles. What would you like to know?"
