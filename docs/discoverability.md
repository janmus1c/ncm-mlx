---
title: Finding and maintaining NCM-MLX documentation
description: Searchable project metadata, static documentation, agent navigation and a minimal process for keeping research claims accurate.
---

# Finding and maintaining the documentation

NCM-MLX uses its code repository as the primary home for source, issues, releases and research evidence. The documentation site renders the same Markdown under `docs/`, with local search and a link to the source of every page. It has no application backend, third-party analytics or tracking cookies. Page titles, summaries, canonical links and the generated sitemap describe actual technical content.

## GitHub discovery

The repository description is:

> Independent MLX implementation of Next-Concept Modeling (NCM): parallel coarse-to-fine text generation, training, and benchmarks.

Topics describe method, framework, hardware and research scope: `mlx`, `apple-silicon`, `next-concept-modeling`, `ncm`, `non-autoregressive`, `coarse-to-fine`, `text-generation`, `vector-quantization`, `language-modeling`, `machine-learning-research`, `cuda` and `python`. All except the specific `next-concept-modeling` topic had exact matches in GitHub topic search at the 2026-10-24 audit; the specific method name is retained because it accurately identifies the project. GitHub permits up to 20 topics; this project does not try to fill the quota. See [GitHub topic guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics).

GitHub's default repository search covers names, descriptions and topics. README contents are searchable using `in:readme`; deeper documentation and symbols need code search or direct links. Examples are `"next-concept" mlx`, `topic:mlx topic:non-autoregressive`, and `"coarse-to-fine" in:readme`. [GitHub search syntax](https://docs.github.com/en/search-github/searching-on-github/searching-for-repositories) documents these distinctions. Code search and Copilot repository indexing are separate systems; public source access alone does not establish inclusion in either index.

## Web and agent access

The static site makes architecture, benchmark methodology and reproduction instructions directly accessible as separate HTML pages. This gives readers stable topic-specific links and a sitemap, rather than requiring a long README for every question. It does not guarantee better ranking than GitHub. [Google's developer guidance](https://developers.google.com/search/docs/fundamentals/get-started-developers) recommends crawlable links, semantic HTML, descriptive titles and sitemaps; indexing remains a search-engine decision.

[Capabilities](capabilities.md), [AGENTS.md](../AGENTS.md), [CITATION.cff](../CITATION.cff) and package metadata describe supported workflows without custom integrations. [`llms.txt`](../llms.txt) is a short navigation map for tools that choose to read it. It follows a [community proposal](https://llmstxt.org/), not a universal agent standard. Google explicitly says special AI text files are unnecessary and do not improve its Search visibility; see its [AI Search guidance](https://developers.google.com/search/docs/fundamentals/ai-optimization-guide). No claim is made about model training, inclusion in assistant answers or recommendation priority.

## Verification and normal indexing delays

After a release, verify the repository, README, tag, documentation, `CITATION.cff`, `llms.txt` and sitemap without authentication. Confirm the archive remains private through authenticated settings. Check representative GitHub repository queries and ordinary web search, recording date, query and actual outcome. A newly public repository may not appear immediately; an empty query result is not proof of a technical fault or a complete view of the index. Do not repeatedly rewrite metadata in response to indexing delay.

The 2026-10-24 preparation audit verified metadata through authenticated readback while the repository was private. Anonymous-access and search observations are recorded in the release completion report after publication. No indexing or ranking claim follows from that configuration check. No Search Console or Bing Webmaster account was connected for this release, so no authenticated sitemap submission is claimed. The sitemap is available for crawlers at the documentation site's `sitemap.xml`.

## Minimal maintenance

At a meaningful public milestone, update supported behavior, the relevant technical page and release notes. Update headline results only from verified artifacts, retain negative results and training-budget caveats, and preserve stable tags/citations. Routine experiments belong in [EXPERIMENTS](EXPERIMENTS.md); they need not trigger broad documentation or topic changes.

Validate locally with:

```bash
uv run python scripts/validate_repository.py
uv run --group docs mkdocs build --strict
```

GitHub Actions rebuilds the site from `main`. Keep source Markdown authoritative and fix broken links before publishing. DOI archival and independently reviewed model cards are useful future research milestones; external posts, artificial links and promotional campaigns are not part of this maintenance process.
