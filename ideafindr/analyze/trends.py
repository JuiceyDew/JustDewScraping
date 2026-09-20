"""Google Trends signal booster.

This is NOT a collector - it doesn't return documents. Instead, it provides
trend validation data that can be used to:
1. Validate momentum findings from social data
2. Add search interest context to reports
3. Identify seasonal patterns

Free tier: Unofficial API, rate limited to ~100 req/day
ToS: Gray area but widely used via pytrends
"""

from __future__ import annotations

import logging
from typing import Any

from ideafindr.models import RunPlan

log = logging.getLogger(__name__)

try:
    from pytrends.trendreq import TrendReq
    
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    log.info("pytrends not installed - Google Trends features disabled")


def get_trend_data(
    keywords: list[str],
    timeframe: str = "today 180-d",
    geo: str = "",
) -> dict[str, Any]:
    """Fetch Google Trends data for keywords.
    
    Args:
        keywords: List of search terms
        timeframe: Time period (e.g., "today 180-d", "today 12-m", "2023-01-01 2024-01-01")
        geo: Geographic region (e.g., "US", "GB", "" for worldwide)
    
    Returns:
        Dict with:
        - interest_over_time: list of (date, value) tuples per keyword
        - interest_by_region: dict of region -> interest score
        - related_queries: dict of keyword -> related queries
        - trend_score: 0-100 overall trend score
    """
    if not PYTRENDS_AVAILABLE:
        return {"error": "pytrends not installed"}
    
    if not keywords:
        return {"error": "no keywords provided"}
    
    try:
        pytrends = TrendReq(hl="en-US", tz=360)
        
        # Build payload
        pytrends.build_payload(
            kw_list=keywords[:5],  # Max 5 keywords per request
            timeframe=timeframe,
            geo=geo,
        )
        
        # Interest over time
        interest_over_time = {}
        try:
            iot = pytrends.interest_over_time()
            if not iot.empty:
                for kw in keywords[:5]:
                    if kw in iot.columns:
                        interest_over_time[kw] = [
                            (str(date), int(value))
                            for date, value in zip(iot.index, iot[kw])
                        ]
        except Exception as e:  # noqa: BLE001
            log.warning("trends: failed to get interest over time: %s", e)
        
        # Interest by region
        interest_by_region = {}
        try:
            ibr = pytrends.interest_by_region()
            if not ibr.empty:
                for kw in keywords[:5]:
                    if kw in ibr.columns:
                        interest_by_region[kw] = {
                            region: int(value) for region, value in ibr[kw].items()
                        }
        except Exception as e:  # noqa: BLE001
            log.warning("trends: failed to get interest by region: %s", e)
        
        # Related queries
        related_queries = {}
        try:
            rq = pytrends.related_queries()
            if rq:
                for kw in keywords[:5]:
                    if kw in rq and rq[kw] is not None:
                        top = rq[kw].get("top", [])
                        if not top.empty:
                            related_queries[kw] = {
                                "top": [
                                    {"query": row["query"], "value": int(row["value"])}
                                    for _, row in top.iterrows()
                                ]
                            }
        except Exception as e:  # noqa: BLE001
            log.warning("trends: failed to get related queries: %s", e)
        
        # Calculate overall trend score
        trend_score = 0
        if interest_over_time:
            all_values = []
            for kw_data in interest_over_time.values():
                all_values.extend([v for _, v in kw_data])
            if all_values:
                # Normalize to 0-100
                avg_interest = sum(all_values) / len(all_values)
                trend_score = int(avg_interest)
        
        return {
            "interest_over_time": interest_over_time,
            "interest_by_region": interest_by_region,
            "related_queries": related_queries,
            "trend_score": trend_score,
            "timeframe": timeframe,
            "geo": geo or "worldwide",
        }
        
    except Exception as e:  # noqa: BLE001
        log.error("trends: failed to fetch data: %s", e)
        return {"error": str(e)}


def validate_momentum(
    social_momentum: float,
    keywords: list[str],
    days: int = 180,
) -> dict[str, Any]:
    """Compare social momentum with Google Trends data.
    
    Args:
        social_momentum: Momentum score from social data (>1 means rising)
        keywords: Keywords to check trends for
        days: Lookback period
    
    Returns:
        Dict with:
        - social_momentum: Original social momentum
        - search_momentum: Momentum from search trends
        - alignment: "aligned" | "divergent" | "neutral"
        - confidence: "high" | "medium" | "low"
    """
    if not keywords:
        return {
            "social_momentum": social_momentum,
            "search_momentum": 0.0,
            "alignment": "neutral",
            "confidence": "low",
            "note": "no keywords provided",
        }
    
    # Convert days to timeframe string
    if days <= 30:
        timeframe = "today 1-m"
    elif days <= 90:
        timeframe = "today 3-m"
    else:
        timeframe = f"today {days}-d"
    
    trends_data = get_trend_data(keywords[:3], timeframe=timeframe)
    
    if "error" in trends_data:
        return {
            "social_momentum": social_momentum,
            "search_momentum": 0.0,
            "alignment": "neutral",
            "confidence": "low",
            "note": trends_data["error"],
        }
    
    # Calculate search momentum from trend data
    search_momentum = 1.0
    interest_over_time = trends_data.get("interest_over_time", {})
    
    if interest_over_time:
        # Get the first keyword's data
        kw_data = list(interest_over_time.values())[0]
        if len(kw_data) >= 4:  # Need at least 4 data points
            # Compare last 25% vs first 25% of time period
            n = len(kw_data)
            quarter = max(1, n // 4)
            
            early_avg = sum(v for _, v in kw_data[:quarter]) / quarter
            recent_avg = sum(v for _, v in kw_data[-quarter:]) / quarter
            
            if early_avg > 0:
                search_momentum = recent_avg / early_avg
            else:
                search_momentum = 1.0 if recent_avg == 0 else 2.0
    
    # Determine alignment
    if social_momentum > 1.5 and search_momentum > 1.5:
        alignment = "aligned"
        confidence = "high"
    elif social_momentum < 0.8 and search_momentum < 0.8:
        alignment = "aligned"
        confidence = "high"
    elif abs(social_momentum - search_momentum) < 0.3:
        alignment = "aligned"
        confidence = "medium"
    else:
        alignment = "divergent"
        confidence = "medium"
    
    return {
        "social_momentum": round(social_momentum, 2),
        "search_momentum": round(search_momentum, 2),
        "alignment": alignment,
        "confidence": confidence,
        "trend_score": trends_data.get("trend_score", 0),
    }


def enhance_report(report_data: dict, plan: RunPlan) -> dict:
    """Add Google Trends context to a report.
    
    Args:
        report_data: Existing report data
        plan: Run plan with topic and keywords
    
    Returns:
        Enhanced report data with trends section
    """
    if not PYTRENDS_AVAILABLE:
        return report_data
    
    # Get trends for main topic and top keywords
    keywords = [plan.topic] + plan.keywords[:4]
    trends_data = get_trend_data(keywords, timeframe=f"today {plan.days}-d")
    
    if "error" not in trends_data:
        report_data["trends"] = {
            "google_trends": trends_data,
            "validation": validate_momentum(
                report_data.get("momentum", 1.0),
                keywords,
                plan.days,
            ),
        }
    
    return report_data
