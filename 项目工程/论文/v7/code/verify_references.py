"""Record publisher metadata for DOI-bearing references without inventing entries."""

import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
papers = json.loads((ROOT / "references/papers.json").read_text())
results = []
for paper in papers:
    record = {"key": paper["key"], "cited_title": paper["title"], "cited_year": paper["year"]}
    if "doi" in paper:
        try:
            response = requests.get("https://api.crossref.org/works/" + paper["doi"],
                                    headers={"User-Agent": "OlistThesisReproducibility/1.0"},
                                    timeout=30)
            response.raise_for_status()
            data = response.json()["message"]
            record.update({"status": "metadata_retrieved", "doi": data["DOI"],
                           "registered_title": data.get("title"),
                           "journal": data.get("container-title"),
                           "published": data.get("published"),
                           "published_print": data.get("published-print"),
                           "volume": data.get("volume"), "issue": data.get("issue"),
                           "pages": data.get("page"),
                           "authors": data.get("author", [])})
        except Exception as e:
            record.update({"status": "retrieval_failed", "error": str(e)})
    else:
        record.update({"status": "web_verification_recorded_separately_not_fetched_by_this_script",
                       "url": paper.get("url"), "note": paper.get("verified", "see source register")})
    results.append(record)
    (ROOT / "references/verification.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(record["key"], record["status"], flush=True)
