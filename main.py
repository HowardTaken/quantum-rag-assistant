import sys
import uuid

from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

from agent import ask, build_agent
from tools import load_vector_store

# Physics answers routinely contain unicode (minus signs, degree/micro symbols, Greek
# letters); the default Windows console codepage (cp1252) can't encode them and would
# crash mid-answer, so force utf-8 output everywhere.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

console = Console(force_terminal=True, legacy_windows=False)

WELCOME = """
# Quantum RAG Agent
**Stack:** ChromaDB · Gemini · LangGraph agent (search_papers, list_available_papers tools, self-critique)
Type your question and press Enter. Type `exit` or `quit` to close.
"""

def main():
    console.print(Markdown(WELCOME))
    console.print(Rule(style="bright_blue"))

    try:
        console.print("[dim]Loading vector store...[/dim]")
        db = load_vector_store()
        console.print(f"[green]{db._collection.count()} vectors ready.[/green]\n")
        agent = build_agent(db=db)
    except RuntimeError as exc:
        console.print(f"[bold red]Startup failed:[/bold red] {exc}")
        return

    thread_id = str(uuid.uuid4())

    while True:
        try:
            question = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        console.print()
        try:
            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                answer = ask(question, agent, thread_id=thread_id)
        except Exception as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            continue

        console.print(Rule(characters="·", style="dim"))
        console.print("[bold green]Agent:[/bold green]")
        console.print(Markdown(answer))
        console.print(Rule(characters="·", style="dim"))
        console.print()

    console.print("\n[dim]Goodbye.[/dim]")

if __name__ == "__main__":
    main()
