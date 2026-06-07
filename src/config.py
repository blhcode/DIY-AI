from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.ollama_utils import normalize_ollama_base_url

COUNTRY_PRESETS: dict[str, dict[str, str]] = {
    "AU": {
        "stores": "Bunnings,Mitre 10,Amazon Australia,Total Tools,Home Timber & Hardware",
        "region": "au-en",
        "locale": "Australia",
        "currency": "AUD",
    },
    "US": {
        "stores": "Home Depot,Lowe's,Menards,Amazon,Harbor Freight",
        "region": "us-en",
        "locale": "United States",
        "currency": "USD",
    },
    "UK": {
        "stores": "B&Q,Screwfix,Wickes,Amazon",
        "region": "uk-en",
        "locale": "United Kingdom",
        "currency": "GBP",
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "diy.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_seconds: int = 120
    default_country: str = "AU"
    available_stores: str = ""
    default_location: str = ""
    diy_port: int = 8780

    @field_validator("ollama_base_url")
    @classmethod
    def fix_ollama_url(cls, v: str) -> str:
        return normalize_ollama_base_url(v)

    @field_validator("default_country")
    @classmethod
    def normalize_country(cls, v: str) -> str:
        return (v or "AU").strip().upper()

    @property
    def country_preset(self) -> dict[str, str]:
        return COUNTRY_PRESETS.get(self.default_country, COUNTRY_PRESETS["AU"])

    @property
    def stores_list(self) -> list[str]:
        raw = self.available_stores.strip()
        if raw:
            return [s.strip() for s in raw.split(",") if s.strip()]
        return [s.strip() for s in self.country_preset["stores"].split(",") if s.strip()]

    @property
    def ddg_region(self) -> str:
        return self.country_preset["region"]

    @property
    def locale_hint(self) -> str:
        return self.country_preset["locale"]

    @property
    def currency(self) -> str:
        return self.country_preset["currency"]


settings = Settings()
