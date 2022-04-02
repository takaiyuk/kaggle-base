import typer

from src.run import RunParams, run_exp

app = typer.Typer(add_completion=False)


@app.command()
def run(
    exp_name: str,
    debug: bool = False,
) -> None:
    run_params = RunParams(
        exp_name=exp_name,
        is_debug=debug,
    )
    run_exp(run_params)


if __name__ == "__main__":
    app()
