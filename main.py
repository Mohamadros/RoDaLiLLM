"""
RoDaLi Seminar - LLM Adversarial Attacker CLI Demo.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from src.models import CustomerRecord
from src.agent_loop import AgenticRedTeamer

console = Console()

SAMPLE_DATASET = [
    CustomerRecord(
        id="CUST-001",
        title="Dr.",
        first_name="Thomas",
        last_name="Becker",
        company="Marini Systems GmbH",
        street="Mainzer Landstraße 180",
        postal_code="60327",
        city="Frankfurt am Main",
        country="Germany",
        email="thomas.becker@googlemail.com",
        phone="+49 69 12345678",
        date_of_birth="1985-04-12",
    ),
    CustomerRecord(
        id="CUST-002",
        title="",
        first_name="Kristina",
        last_name="Bäcker",
        company="Deutsche Bank AG",
        street="Taunusanlage 12",
        postal_code="60325",
        city="Frankfurt am Main",
        country="Germany",
        email="k.baecker@gmail.com",
        phone="+49 69 98765432",
        date_of_birth="1992-11-23",
    ),
    CustomerRecord(
        id="CUST-003",
        title="Prof. Dr.",
        first_name="Max",
        last_name="Mustermann",
        company="Siemens Energy AG",
        street="Otto-Hahn-Ring 6",
        postal_code="81739",
        city="München",
        country="Germany",
        email="max.mustermann@siemens.de",
        phone="+49 89 5556677",
        date_of_birth="1978-08-30",
    ),
]


def run_benchmark():
    console.print(
        Panel.fit(
            "[bold cyan]RoDaLi Seminar - Adversarial LLM Red-Teamer[/bold cyan]\n"
            "[yellow]Objective:[/yellow] Evolving synthetic duplicates to evade Marini Systems' Record Linkage engine",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    red_teamer = AgenticRedTeamer(evasion_threshold=0.75, max_iterations=3)

    summary_table = Table(
        title="Adversarial Attack Results vs. Linkage Engine (Threshold: 0.75)",
        box=box.HEAVY_EDGE,
        header_style="bold magenta",
    )
    summary_table.add_column("ID", style="dim", width=10)
    summary_table.add_column("Original Identity", width=26)
    summary_table.add_column("Generated Adversarial Identity", width=28)
    summary_table.add_column("Baseline Score", justify="center", width=14)
    summary_table.add_column("Attacked Score", justify="center", width=14)
    summary_table.add_column("Rounds", justify="center", width=8)
    summary_table.add_column("Result", justify="center", width=16)

    for record in SAMPLE_DATASET:
        console.print(f"\n[bold blue]>>> Attacking entity:[/bold blue] {record.first_name} {record.last_name} ({record.company})...")
        attack_result = red_teamer.attack_record(record)

        orig_str = f"{record.first_name} {record.last_name}\n[dim]{record.company}[/dim]\n[dim]{record.street}, {record.city}[/dim]"
        adv = attack_result["final_adversarial_record"]
        adv_str = f"{adv['first_name']} {adv['last_name']}\n[dim]{adv['company']}[/dim]\n[dim]{adv['street']}, {adv['city']}[/dim]"

        baseline_str = f"{attack_result['baseline_score']:.2f}"
        final_str = f"{attack_result['final_score']:.2f}"
        rounds_str = str(attack_result["total_rounds"])

        if attack_result["evasion_successful"]:
            status_str = "[bold green]EVADED (Won)[/bold green]"
        else:
            status_str = "[bold red]CAUGHT[/bold red]"

        summary_table.add_row(
            record.id,
            orig_str,
            adv_str,
            baseline_str,
            final_str,
            rounds_str,
            status_str,
        )

        # Print iteration steps
        for step in attack_result["history"]:
            flag = "[green]✓ EVADED[/green]" if step["evasion_successful"] else "[red]✗ CAUGHT[/red]"
            console.print(
                f"  Round {step['iteration']}: Strategy: [yellow]{step['strategy']}[/yellow] "
                f"-> Similarity Score: [bold]{step['similarity_score']:.2f}[/bold] {flag} "
                f"(Blocking: {'Match' if step['blocking_intact'] else 'BYPASSED'})"
            )

    console.print("\n")
    console.print(summary_table)
    console.print(
        "\n[bold green]Ready for testing![/bold green] "
        "Edit [bold cyan].env[/bold cyan] in VS Code to supply your OPENAI_API_KEY or GEMINI_API_KEY and Marini API credentials."
    )


if __name__ == "__main__":
    run_benchmark()
