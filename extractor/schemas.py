"""Pydantic schemas for the contract extractor.

The 12 fields in :class:`ContractExtraction` are a subset of CUAD's 41 clause
categories, selected for commercial relevance. Field declaration order is
canonical and is relied on by:

- ``training/prepare_dataset.py`` for deterministic JSON serialization of the
  assistant's training target.
- ``evaluation/metrics.py`` for iterating over fields when computing
  ``overall_f1``.

Do not reorder fields without updating those call sites.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ContractExtraction(BaseModel):
    """The structured output schema for contract extraction."""

    # Identity
    document_name: Optional[str] = Field(
        None,
        description="The title or name of the contract document",
    )
    parties: List[str] = Field(
        default_factory=list,
        description="All named parties to the contract",
    )

    # Dates
    agreement_date: Optional[str] = Field(
        None,
        description="Date the agreement was signed, ISO format YYYY-MM-DD if possible",
    )
    effective_date: Optional[str] = Field(
        None,
        description="Date the agreement becomes effective",
    )
    expiration_date: Optional[str] = Field(
        None,
        description="Date the agreement expires, if specified",
    )

    # Legal framework
    governing_law: Optional[str] = Field(
        None,
        description="State, country, or jurisdiction whose law governs the contract",
    )

    # Term and renewal
    renewal_term: Optional[str] = Field(
        None,
        description="How the agreement renews (auto-renewal terms, manual, etc.)",
    )
    notice_period_to_terminate_renewal: Optional[str] = Field(
        None,
        description="Required notice period to prevent automatic renewal",
    )

    # Commercial terms
    exclusivity: Optional[str] = Field(
        None,
        description="Any exclusivity clauses (territorial, customer, product, etc.)",
    )
    non_compete: Optional[str] = Field(
        None,
        description="Any non-compete restrictions",
    )

    # Risk allocation
    cap_on_liability: Optional[str] = Field(
        None,
        description="Maximum liability amount or formula, if capped",
    )
    uncapped_liability: Optional[str] = Field(
        None,
        description="Carve-outs from any cap on liability (e.g., gross negligence, willful misconduct, IP infringement, confidentiality breach)",
    )


class ExtractRequest(BaseModel):
    """Request body for the /extract endpoint."""

    contract_text: str = Field(
        ...,
        min_length=50,
        description="The full text of the contract to extract from",
    )


class ExtractResponse(BaseModel):
    """Response from the /extract endpoint."""

    extraction: ContractExtraction
    inference_time_ms: float
    tokens_generated: int
