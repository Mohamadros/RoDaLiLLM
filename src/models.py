"""
Data models for customer records in the RoDaLi benchmark.
"""

from typing import Optional
from pydantic import BaseModel, Field


class CustomerRecord(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = Field(default="", description="Academic or courtesy title (Dr., Prof., Herr, Frau)")
    first_name: str = Field(description="First/given name")
    last_name: str = Field(description="Last/family name")
    company: Optional[str] = Field(default="", description="Company or organization name")
    street: str = Field(description="Street address and building number")
    postal_code: str = Field(description="ZIP or postal code")
    city: str = Field(description="City or town")
    country: str = Field(default="Germany", description="Country")
    email: Optional[str] = Field(default="", description="Email address")
    phone: Optional[str] = Field(default="", description="Phone number")
    date_of_birth: Optional[str] = Field(default="", description="Date of birth in standard format")

    def to_full_text(self) -> str:
        """Serializes the record into a unified string representation for string matching."""
        parts = [
            self.title,
            self.first_name,
            self.last_name,
            self.company,
            self.street,
            self.postal_code,
            self.city,
            self.country,
            self.email,
            self.phone,
            self.date_of_birth,
        ]
        return " ".join([p.strip() for p in parts if p and p.strip()])

    def blocking_key(self) -> str:
        """
        Common standard blocking key used in commercial linkage pipelines:
        First 3 chars of last name + postal code.
        """
        last_clean = "".join(filter(str.isalpha, self.last_name)).upper()
        last_prefix = last_clean[:3].ljust(3, "X")
        postal_clean = "".join(filter(str.isalnum, self.postal_code))
        return f"{last_prefix}_{postal_clean}"
