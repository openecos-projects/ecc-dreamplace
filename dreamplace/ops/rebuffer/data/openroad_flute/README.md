# OpenROAD FLUTE LUT provenance

These tables are copied from OpenROAD `src/stt/src/flt/etc` at commit
`7201893c561454a9bb1245b5a449bb1340801f17`.

```text
POST9.dat  sha256 20455530e0eb52c5b0b0405e3ae25c20f9f1e9af00e2a70290136e1243f94d01
POWV9.dat  sha256 876eb7fd9b6e2f98fc4e5679545f7d563cf32dd44fd4b2c4b97893066bb3b699
```

OpenROAD's build encodes these files as base64 string literals.  The native
provider reads the same bytes directly so its FLUTE result stays identical
without a generated 12 MB C++ source file.

License: BSD-3-Clause, see `LICENSE`.
