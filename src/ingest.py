from __future__ import annotations

from pathlib import Path

from pulso_transmi import PulsoTransmiClient


RAW_FILENAMES = ("stations.csv", "observations.csv", "context.csv", "metadata.json")


def download_raw_data(output_dir: Path = Path("data/raw")) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    with PulsoTransmiClient() as client:
        metadata = client.meta()["dataset"]
        print(
            f"Dataset {metadata['dataset']}: {metadata['station_count']} estaciones, "
            f"{metadata['observation_rows']:,} observaciones, "
            f"{metadata['context_rows']:,} registros de contexto"
        )
        for filename in RAW_FILENAMES:
            path = client.download(filename, output_dir / filename)
            downloaded.append(path)
            print(f"descargado: {path}")

    return downloaded


def main() -> None:
    download_raw_data()


if __name__ == "__main__":
    main()
