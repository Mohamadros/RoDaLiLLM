"""
Data model and parser for the CDDB (Compact Disc Database) benchmark dataset.
"""

from typing import List, Optional, Dict, Any
import xml.etree.ElementTree as ET
from pydantic import BaseModel, Field
import jellyfish
from Levenshtein import ratio as levenshtein_ratio


class CDDBRecord(BaseModel):
    did: str = Field(description="CDDB Disc ID")
    artist: str = Field(description="Album Artist or Band")
    dtitle: str = Field(description="Disc / Album Title")
    year: Optional[str] = Field(default="", description="Release Year")
    genre: Optional[str] = Field(default="", description="Music Genre")
    category: Optional[str] = Field(default="", description="CDDB Category")
    tracks: List[str] = Field(default_factory=list, description="List of track titles")

    def blocking_key(self) -> str:
        """
        Standard blocking key for CDDB albums:
        First 3 alphanumeric characters of artist + first 3 characters of album title.
        """
        clean_artist = "".join(filter(str.isalnum, self.artist)).upper()
        prefix_artist = clean_artist[:3].ljust(3, "X")
        clean_title = "".join(filter(str.isalnum, self.dtitle)).upper()
        prefix_title = clean_title[:3].ljust(3, "X")
        return f"{prefix_artist}_{prefix_title}"

    def to_full_text(self) -> str:
        track_str = " | ".join(self.tracks[:5])
        return f"{self.artist} - {self.dtitle} ({self.year}) [{self.genre}] Tracks: {track_str}".strip()


def compute_cddb_similarity(rec1: CDDBRecord, rec2: CDDBRecord, threshold: float = 0.75) -> Dict[str, Any]:
    """
    Computes duplicate detection similarity between two CDDB album records:
    - Artist: Jaro-Winkler
    - Album Title: Levenshtein
    - Year: Exact match
    - Tracks: Average Jaccard & Jaro-Winkler across top tracks
    """
    bk1 = rec1.blocking_key()
    bk2 = rec2.blocking_key()
    blocking_match = (bk1 == bk2)

    # 1. Artist similarity
    artist_sim = jellyfish.jaro_winkler_similarity(rec1.artist.lower(), rec2.artist.lower())

    # 2. Album title similarity
    title_sim = levenshtein_ratio(rec1.dtitle.lower(), rec2.dtitle.lower())

    # 3. Year similarity
    year_sim = 1.0 if rec1.year and rec2.year and rec1.year.strip() == rec2.year.strip() else 0.5 if not rec1.year or not rec2.year else 0.0

    # 4. Tracks overlap (Jaccard on combined track words)
    t1 = set(" ".join(rec1.tracks).lower().split())
    t2 = set(" ".join(rec2.tracks).lower().split())
    if t1 and t2:
        track_sim = len(t1 & t2) / len(t1 | t2)
    else:
        track_sim = 0.5

    # Weighted composite score
    weights = {"artist": 0.40, "title": 0.35, "tracks": 0.15, "year": 0.10}
    composite_score = (
        weights["artist"] * artist_sim +
        weights["title"] * title_sim +
        weights["tracks"] * track_sim +
        weights["year"] * year_sim
    )

    # Decision: duplicate match requires passing blocking and exceeding the threshold
    is_match = (blocking_match and composite_score >= threshold)
    evaded = not is_match

    return {
        "blocking_key_1": bk1,
        "blocking_key_2": bk2,
        "blocking_match": blocking_match,
        "composite_score": round(composite_score, 4),
        "artist_sim": round(artist_sim, 4),
        "title_sim": round(title_sim, 4),
        "track_sim": round(track_sim, 4),
        "threshold": threshold,
        "is_match": is_match,
        "evaded": evaded,
    }


def load_sample_cddb_records(xml_file: str, sample_size: int = 10) -> List[CDDBRecord]:
    """Efficiently parses sample records from the CDDB XML file."""
    records = []
    for event, elem in ET.iterparse(xml_file, events=('end',)):
        if elem.tag == 'disc':
            artist = elem.findtext('artist', '').strip()
            dtitle = elem.findtext('dtitle', '').strip()
            tracks = [t.text.strip() for t in elem.findall('tracks/title') if t.text]
            
            # Select readable records with Latin characters and tracks
            if artist and dtitle and len(tracks) >= 3 and all(ord(c) < 256 for c in artist) and any(c.isalpha() for c in artist):
                rec = CDDBRecord(
                    did=elem.findtext('did', ''),
                    artist=artist,
                    dtitle=dtitle,
                    year=elem.findtext('year', '').strip(),
                    genre=elem.findtext('genre', '').strip(),
                    category=elem.findtext('category', '').strip(),
                    tracks=tracks[:8],
                )
                records.append(rec)
                if len(records) >= sample_size:
                    break
            elem.clear()
    return records
