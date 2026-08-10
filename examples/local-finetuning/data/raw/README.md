# Immutable raw inputs

`make prepare-flight` downloads the exact Kaggle archive here and verifies its
SHA-256 digest before extracting it. Do not edit downloaded files. Refreshing to
a later dataset version is a reviewed data change: update the manifest, card,
processing configuration, and frozen split together.

The pinned Kaggle CLI syntax used by preparation is:

```bash
uv run kaggle datasets download \
  bitext/bitext-gen-ai-chatbot-customer-support-dataset \
  --path data/raw/bitext
```

The public download was anonymous when this project was verified. If Kaggle
later requires authentication, configure a currently supported Kaggle
credential outside this directory; never add it to Git. If automated download
is unavailable, download the archive manually from the URL in
`configs/project.yaml`, name it exactly as configured under `data/raw/bitext/`,
and rerun `make prepare-flight`. The preparation gate rejects any archive or
CSV whose SHA-256 differs from the reviewed version.
