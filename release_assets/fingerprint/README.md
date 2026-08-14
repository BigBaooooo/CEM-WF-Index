# Background-context fingerprint archive

The GitHub Release asset `cem_wf_context_fingerprints_1979_2020.zip`
contains precomputed ERA5 background-context fingerprints used by the public
`FingerprintArchive` and `ContextIndex` interfaces. The arrays are supplied as
an upstream retrieval input; they are not trained by the frozen final ranker.

After extracting the asset, verify it from this directory:

```bash
sha256sum -c SHA256SUMS
```

The archive contains 368,161 overlapping 24-hour windows at one-hour stride.
Each 1,024-dimensional row concatenates four L2-normalized 256-dimensional
blocks in this order: `t2m`, `msl`, `u10`, `v10`. See `manifest.json` for the
complete public schema and `ATTRIBUTION.txt` for ERA5 attribution.

The archive is distributed as a version-neutral, checksum-verified input to the
public background-context retrieval interface.
