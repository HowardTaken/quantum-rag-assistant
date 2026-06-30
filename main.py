from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
from rag_agent import load_vector_store, build_chain, ask

console = Console(force_terminal=True, legacy_windows=False)

WELCOME = """
# Quantum RAG Agent
**Stack:** ChromaDB · Gemini Embeddings · gemini-2.5-flash
Type your question and press Enter. Type `exit` or `quit` to close.
"""

def main():
    console.print(Markdown(WELCOME))
    console.print(Rule(style="bright_blue"))

    console.print("[dim]Loading vector store...[/dim]")
    db = load_vector_store()
    console.print(f"[green]{db._collection.count()} vectors ready.[/green]\n")

    chain = build_chain(db)

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
        with console.status("[dim]Thinking...[/dim]", spinner="dots"):
            answer = ask(question, chain)

        console.print(Rule(characters="·", style="dim"))
        console.print("[bold green]Agent:[/bold green]")
        console.print(Markdown(answer))
        console.print(Rule(characters="·", style="dim"))
        console.print()

    console.print("\n[dim]Goodbye.[/dim]")

if __name__ == "__main__":
    main()
