# RAG Evaluation Results

## Config
| Setting | Value |
|---|---|
| Language | python |
| Repos tested | 1 |
| Queries evaluated | 50 |
| Top-K | 5 |
| Reranker | False |

## Metrics
| Metric | Score |
|---|---|
| Recall@5 | 0.780 |
| MRR | 0.646 |
| Hits | 39/50 |
| Errors | 0 |

## Repos Tested
- https://github.com/soimort/you-get

## Sample Failures (first 10)
| Query | Expected File | Top Retrieved |
|---|---|---|
| wrapper... | https://github.com/soimort/you-get/blob/b746ac01c9f39de94cac2d56f665285b0523b974/src/you_get/extractors/fc2video.py#L46-L57 | .github/workflows/python-package.yml |
| http://stackoverflow.com/a/30923963/2946714... | https://github.com/soimort/you-get/blob/b746ac01c9f39de94cac2d56f665285b0523b974/src/you_get/extractors/ucas.py#L18-L29 | CHANGELOG.rst |
| wrapper... | https://github.com/soimort/you-get/blob/b746ac01c9f39de94cac2d56f665285b0523b974/src/you_get/extractors/yixia.py#L65-L93 | .github/workflows/python-package.yml |
| Source: Android mobile... | https://github.com/soimort/you-get/blob/b746ac01c9f39de94cac2d56f665285b0523b974/src/you_get/extractors/veoh.py#L19-L33 | .github/workflows/python-package.yml |
| str->None... | https://github.com/soimort/you-get/blob/b746ac01c9f39de94cac2d56f665285b0523b974/src/you_get/extractors/vimeo.py#L15-L19 | src/you_get/extractors/netease.py |
| str/int->None... | https://github.com/soimort/you-get/blob/b746ac01c9f39de94cac2d56f665285b0523b974/src/you_get/extractors/vimeo.py#L22-L36 | src/you_get/util/strings.py |
| Splicing URLs according to video ID to get video details... | https://github.com/soimort/you-get/blob/b746ac01c9f39de94cac2d56f665285b0523b974/src/you_get/extractors/ixigua.py#L34-L78 | src/you_get/extractors/youtube.py |
| Extracts video ID from URL.... | https://github.com/soimort/you-get/blob/b746ac01c9f39de94cac2d56f665285b0523b974/src/you_get/extractors/mgtv.py#L27-L33 | src/you_get/extractors/mgtv.py |
| Override the original one
        Ugly ugly dirty hack... | https://github.com/soimort/you-get/blob/b746ac01c9f39de94cac2d56f665285b0523b974/src/you_get/extractors/iqiyi.py#L158-L218 | src/you_get/extractors/missevan.py |
| str, str, str, bool, bool ->None

    Download Acfun video b... | https://github.com/soimort/you-get/blob/b746ac01c9f39de94cac2d56f665285b0523b974/src/you_get/extractors/acfun.py#L42-L109 | src/you_get/extractors/acfun.py |
