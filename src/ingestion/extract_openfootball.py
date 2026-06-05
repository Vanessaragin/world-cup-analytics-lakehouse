from pathlib import Path
import requests


YEARS = WORLD_CUP_YEARS = [
    1930,1934,1938,1950,1954,1958,1962,1966,1970,1974,1978,1982,
    1986,1990,1994,1998,2002, 2006, 2010,2014, 2018, 2022,2026
    ]

RAW_DIR = Path("data/raw/openfootball")


def build_url(year: int) -> str:
    return f"https://raw.githubusercontent.com/openfootball/worldcup.json/master/{year}/worldcup.json"


def download_file(year: int):

    url = build_url(year)

    output_file = RAW_DIR / f"worldcup_{year}_raw.json"

    response = requests.get(url, timeout=30)

    if response.status_code != 200:
        print(f"Erro ao baixar {year}")
        return

    output_file.write_text(
        response.text,
        encoding="utf-8"
    )

    print(f"Download concluído: {year}")


if __name__ == "__main__":

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for year in YEARS:
        download_file(year)