#!/bin/bash
python3 surgery.py ready_trader_one/match_events.csv img/pricing.pdf $1 $2
python3 surgery.py logs/round1/match31_events.csv img/pricing_31.pdf $1 $2
