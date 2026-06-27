"""AuctionSense — Pydantic schemas for FastAPI request/response validation."""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class LotCategory(str, Enum):
    ferrous_scrap          = "ferrous_scrap"
    nonferrous_scrap       = "nonferrous_scrap"
    idle_industrial_asset  = "idle_industrial_asset"
    coal                   = "coal"
    agricultural           = "agricultural"


class LotGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class AuctionFormat(str, Enum):
    english_ascending  = "english_ascending"
    sealed_bid         = "sealed_bid"
    dutch_descending   = "dutch_descending"


class LocationRegion(str, Enum):
    east  = "east"
    west  = "west"
    north = "north"
    south = "south"


class OperatorSector(str, Enum):
    commercial_scrap        = "commercial_scrap"
    government_liquidation  = "government_liquidation"
    agricultural_commodity  = "agricultural_commodity"
    coal_resources          = "coal_resources"
    real_estate             = "real_estate"


class LotInput(BaseModel):
    lot_category:            LotCategory    = Field(..., description="Commodity category of the lot")
    lot_quantity_mt:         float          = Field(..., gt=0, description="Lot volume in metric tonnes")
    lot_grade:               LotGrade       = Field(..., description="Quality grade A (best) to D (lowest)")
    lot_location_region:     LocationRegion = Field(..., description="Geographic zone of the lot")
    days_to_auction:         int            = Field(..., ge=1, le=30, description="Days until auction event")
    auction_format:          AuctionFormat  = Field(AuctionFormat.english_ascending, description="Auction format")
    registered_buyer_count:  int            = Field(..., ge=0, description="Registered buyers in this category")
    season_quarter:          str            = Field("Q1", pattern="^Q[1-4]$", description="Current quarter")

    model_config = {
        "json_schema_extra": {
            "example": {
                "lot_category": "ferrous_scrap",
                "lot_quantity_mt": 150.0,
                "lot_grade": "B",
                "lot_location_region": "east",
                "days_to_auction": 3,
                "auction_format": "english_ascending",
                "registered_buyer_count": 47,
                "season_quarter": "Q1",
            }
        }
    }


class BriefingRequest(BaseModel):
    lot: LotInput
    operator_sector: OperatorSector = OperatorSector.commercial_scrap

    model_config = {
        "json_schema_extra": {
            "example": {
                "lot": {
                    "lot_category": "ferrous_scrap",
                    "lot_quantity_mt": 150.0,
                    "lot_grade": "B",
                    "lot_location_region": "east",
                    "days_to_auction": 3,
                    "auction_format": "english_ascending",
                    "registered_buyer_count": 47,
                    "season_quarter": "Q1",
                },
                "operator_sector": "commercial_scrap"
            }
        }
    }


class BriefingResponse(BaseModel):
    lot_category:           str
    pre_auction_briefing:   str             = Field(..., description="Full Pre-Auction Briefing text")
    reserve_result:         dict[str, Any]  = Field(..., description="Reserve Price Agent output")
    participation_result:   dict[str, Any]  = Field(..., description="Participation Agent output")
    lot_config_result:      dict[str, Any]  = Field(..., description="Lot Configuration Agent output")
    compliance_flags:       list[str]       = Field(default_factory=list)
    pipeline_errors:        list[str]       = Field(default_factory=list)
    latency_seconds:        float


class HealthResponse(BaseModel):
    status:  str
    version: str
