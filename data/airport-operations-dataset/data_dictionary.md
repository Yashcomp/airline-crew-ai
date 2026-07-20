# Airport Operations Dataset — Data Dictionary

This dataset simulates airport operations at Indira Gandhi International Airport (DEL).

Tables included:

1. flights → flight schedule, delays, aircraft and route info
2. passengers → passenger demographics and journey information
3. baggage → baggage handling lifecycle
4. gate_events → boarding and gate movement events
5. security_screening → passenger security processing
6. staff_shifts → workforce scheduling and assignment
7. retail_transactions → airport retail purchases
8. maintenance_logs → aircraft engineering and defect tracking

Relationships:

- flight_id links flights → passengers → baggage → gate_events → retail → maintenance
- passenger_id and pnr_code link passenger journey tables
- staff_id links operational workforce records

Possible use-cases:

- Flight delay prediction
- Passenger flow analytics
- Security queue optimization
- Retail revenue forecasting
- Maintenance reliability modelling
- Airport operations dashboarding