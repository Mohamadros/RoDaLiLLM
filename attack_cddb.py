"""
RoDaLi Seminar - CDDB Benchmark Adversarial Attack Runner.
Attacks real music album records from data/cddb_ID_nested_10000.xml using Gemini LLM.
"""

import os
import json
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from google import genai
from google.genai import types

from src.cddb_models import CDDBRecord, load_sample_cddb_records, compute_cddb_similarity
from src.perturbations import apply_ocr_glitch, apply_keyboard_typo

load_dotenv()
console = Console()

CDDB_PROMPT_TEMPLATE = """You are an adversarial synthetic data generator for an academic Record Linkage / Entity Resolution benchmark (RoDaLi seminar).
You are given a ground-truth music CD / album record from the CDDB benchmark.
Your goal is to rewrite it into an adversarial duplicate representing the EXACT SAME album, but formatted to evade duplicate detection algorithms.

GROUND TRUTH RULES:
1. Semantic Identity: It must unambiguously represent the same album.
2. Attack Techniques:
   - Artist variations: Move or drop leading articles (e.g. "The Cranberries" -> "Cranberries, The" or "Cranberries"), add featured artists (e.g. "Eminem" -> "Eminem feat. Dido").
   - Title variations: Add subtle edition/release tags (e.g. "[2004 Remaster]", "(Deluxe Edition)", "(Bonus Track Version)", roman numerals "Vol. IV" vs "Vol. 4").
   - Tracks: Add realistic suffixes like "(Radio Mix)", "(Album Version)", "(Live)", or reword slightly.
   - Genre: Use synonymous subgenres (e.g. "Rock" -> "Alternative Rock", "Electronic" -> "Trance / Dance").

Input CDDB Record:
{input_json}

Return ONLY a valid JSON object matching the exact schema:
{{
  "did": "{did}",
  "artist": "...",
  "dtitle": "...",
  "year": "...",
  "genre": "...",
  "category": "...",
  "tracks": ["..."]
}}
"""


def attack_cddb_with_gemini(record: CDDBRecord, client: genai.Client, model_name: str) -> CDDBRecord:
    """Uses Gemini LLM to semantically perturb a CDDB record with retry logic."""
    import time
    prompt = CDDB_PROMPT_TEMPLATE.format(
        input_json=record.model_dump_json(indent=2),
        did=record.did,
    )
    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.7)
    
    for attempt in range(2):
        try:
            resp = client.models.generate_content(model=model_name, contents=prompt, config=cfg)
            raw = resp.text.strip()
            data = json.loads(raw)
            return CDDBRecord(**data)
        except Exception as e:
            if attempt == 0:
                time.sleep(1.0)
                continue
            console.print(f"[yellow]LLM notice ({e}). Using rule-based fallback.[/yellow]")
            # Fallback heuristic variation
            art = record.artist
            if art.lower().startswith("the "):
                art = art[4:] + ", The"
            elif " " in art:
                art = art.lower()
            title = f"{record.dtitle} (Remastered)"
            return CDDBRecord(
                did=record.did,
                artist=art,
                dtitle=title,
                year=record.year,
                genre=record.genre,
                category=record.category,
                tracks=[t + " (Album Version)" for t in record.tracks],
            )


def main():
    xml_file = "data/cddb_ID_nested_10000.xml"
    if not os.path.exists(xml_file):
        console.print(f"[red]Error: Dataset not found at {xml_file}[/red]")
        return

    console.print(
        Panel.fit(
            "[bold cyan]RoDaLi Seminar - CDDB Dataset Adversarial Attack[/bold cyan]\n"
            f"[yellow]Dataset:[/yellow] {xml_file} (9,763 albums)\n"
            "[yellow]Objective:[/yellow] Generating adversarial duplicates to break CDDB entity resolution",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    client = genai.Client(api_key=api_key)

    console.print("[dim]Loading sample albums from XML dataset...[/dim]")
    records = load_sample_cddb_records(xml_file, sample_size=4)

    table = Table(
        title="CDDB Entity Resolution Evasion Results (Threshold: 0.75)",
        box=box.HEAVY_EDGE,
        header_style="bold magenta",
    )
    table.add_column("Disc ID", style="dim", width=10)
    table.add_column("Original Album", width=30)
    table.add_column("Adversarial Duplicate (Gemini)", width=32)
    table.add_column("Similarity", justify="center", width=12)
    table.add_column("Blocking", justify="center", width=12)
    table.add_column("Result", justify="center", width=16)

    for rec in records:
        console.print(f"\n[bold blue]>>> Attacking album:[/bold blue] [cyan]{rec.artist}[/cyan] - [green]{rec.dtitle}[/green]")
        
        # Round 1: Gemini semantic perturbation
        adv_rec = attack_cddb_with_gemini(rec, client, model_name)

        # Check similarity
        sim_res = compute_cddb_similarity(rec, adv_rec, threshold=0.75)

        # Round 2: Stack subtle OCR / typo perturbation if still caught
        if sim_res["is_match"]:
            console.print("  [yellow]Round 1 caught (score >= 0.75). Stacking OCR noise on track and title...[/yellow]")
            adv_rec.dtitle = apply_ocr_glitch(adv_rec.dtitle)
            adv_rec.artist = apply_keyboard_typo(adv_rec.artist, error_rate=0.1, target_prefix=True)
            sim_res = compute_cddb_similarity(rec, adv_rec, threshold=0.75)

        blocking_str = "[green]Match[/green]" if sim_res["blocking_match"] else "[bold red]BYPASSED[/bold red]"
        score_str = f"{sim_res['composite_score']:.2f}"
        status_str = "[bold green]EVADED (Won)[/bold green]" if sim_res["evaded"] else "[bold red]CAUGHT[/bold red]"

        orig_summary = f"{rec.artist}\n[dim]{rec.dtitle} ({rec.year or 'N/A'})[/dim]\n[dim]Tracks: {len(rec.tracks)}[/dim]"
        adv_summary = f"{adv_rec.artist}\n[dim]{adv_rec.dtitle} ({adv_rec.year or 'N/A'})[/dim]\n[dim]Tracks: {len(adv_rec.tracks)}[/dim]"

        table.add_row(
            rec.did,
            orig_summary,
            adv_summary,
            score_str,
            blocking_str,
            status_str,
        )

        console.print(
            f"  -> Composite Similarity: [bold]{sim_res['composite_score']:.2f}[/bold] "
            f"(Artist: {sim_res['artist_sim']:.2f}, Title: {sim_res['title_sim']:.2f}, Tracks: {sim_res['track_sim']:.2f}) "
            f"| Blocking: {blocking_str} | Status: {status_str}"
        )

    console.print("\n")
    console.print(table)
    console.print(
        "\n[bold green]Success![/bold green] Real CDDB benchmark dataset records have been attacked and evaluated."
    )


if __name__ == "__main__":
    main()
