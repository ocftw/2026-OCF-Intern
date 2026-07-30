# Version pinning evidence

## Dataset

The Hugging Face official repository history identifies:

- Revision: `d386947f7fc3bafdcd756c8485845a2f43a19875`
- Commit title: `add v1.6`
- Date: 2026-04-10
- `OmniDocBench.json` size: 42,208,096 bytes
- SHA-256: `a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`
- Parsed pages: 1,651
- Unique image filenames: 1,651
- Subsets: 1,355 v1.5 base + 100 equation_hard + 99 layout_hard +
  97 table_hard

The revision and hash are immutable configuration fields. A current-main
download, v1.7 annotation, 1,355-page base file, or mismatching hash blocks the
pipeline.

## Evaluation code

Official Git history shows `147cd5ac9472002f5751221d390bf00abdbc0d2f`
(`update README to v1.6`, 2026-04-10) as the last v1.6 main commit before the
2026-04-30 v1.7 update. The official tree contains no Gemma inference script or
universal end-to-end VLM prompt, so the common user-specified prompt is used and
hashed.

## Docker reproduction image

- RepoDigest and image ID:
  `sha256:6116ad72172e763b5c43e963d5efebf2093f2362b975f58156ce4f6c9142e617`
- Python 3.10.16
- TeX Live 2025 / pdfTeX 3.141592653-2.6-1.40.28
- ImageMagick 7.1.1-47
- Ghostscript 9.55.0
- Supervisor config SHA-256:
  `e15568889c314beca1fe511103dcdf6612af61958ccb13a9e2ae922bd7c31880`

Important discrepancy: the image-embedded file
`/workspace/gt/OmniDocBench_v1.6_base.json` has SHA-256
`f3352b65ca4c0b7cf02844eda3de1fd1b2cad1be359d7aeeb4c2f18d34915b16`
and only 1,355 pages. It is the base subset, not v1.6 full. The runner never
uses it for this benchmark; it mounts the externally pinned 1,651-page GT
read-only.

The supervisor reproduction config requests `quick_match`, text Edit_dist,
formula Edit_dist+CDM, table TEDS+Edit_dist, and reading-order Edit_dist. Its
hard-coded worker count is 13. The immutable run copy uses 4/3/3 workers for
match/CDM/TEDS on this 8-vCPU, 61-GB host, preserving metrics and matching
method while preventing oversubscription.
