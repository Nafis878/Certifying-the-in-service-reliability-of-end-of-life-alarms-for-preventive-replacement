# Path redaction in the frozen code

After the protocol was frozen, the absolute working-directory paths in seven of these files were replaced by the
placeholders `<WORKDIR>` (the analysis working folder) and `<STUDY_REPO>` (the study pipeline, `study/` in this
repository), for anonymity. Only the lines listed below changed; every other byte is as hashed. These seven files
therefore no longer match their entries in `../SHA256SUMS`, which is kept unchanged as written at freezing (its own
hash is still the one quoted in the top-level README). The other frozen files still match.

| File | Lines changed | SHA-256 of the published copy |
|---|---|---|
| `e1_analyze.py` | 4 | `af63a2274b34bd346f6c4a94ee56e052bc2977015def709241ff741eb52c87c5` |
| `e1_fit.py` | 3,5 | `3e58d82ead139d9ba3403abe31b4f3a20a78ac0bbab3866d31da084b8c3a0ccf` |
| `e2_analyze.py` | 3 | `787de2b7b31eea9af96b34ccb49b5c3e2fff4b3c8b5855fe51bfefb25f672ce8` |
| `e2_fit.py` | 3,6 | `5a10a69e9efda3df9b62201a1871bcd7c16df35125af7b44b6daa118bbc143dc` |
| `e3_e6_analyze.py` | 4 | `1b3955d7156afbad0a4e3c845fefbd81f478ece16d996ee5e0cd8f0b4f1fbc8a` |
| `recipe.py` | 4 | `674ee2e2aa37d9b8fc71f2f860cecee0a8c269245391c0ebc146c11c72f90de4` |
| `repro_check.py` | 1 | `3600c5cba478b76817ce5d4bba5554a36113272de8394136e955e17f059deb78` |

To run these copies, replace the placeholders with your own paths; the maintained versions in `new_analyses/code`
use relative paths.
