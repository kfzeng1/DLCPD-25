# IP102 Detection Data Handoff v1

Status: rebuilt as the Plan-A detection data contract after the repository reorganisation on 2026-08-17. The official IP102 Detection VOC partition and cleaning rules are unchanged from the accepted v1 contract.

## Inputs and policy

- Raw root: `data/raw/ip102/VOC2007/` (read-only)
- Official trainval/test: 15,178/3,798; official test is preserved exactly and was not used for split or cleaning decisions.
- Derived train/val: 12,142/3,036, deterministic iterative multilabel stratification v1, seed 20260812, validation ratio 0.20.
- All paths in annotations are relative to the VOC root. The five extra JPEGs are listed in `exceptions.json` and excluded.

## Annotation and label contract

- `annotations.jsonl` contains all 18,976 formal IDs, 22,284 traceable raw boxes and 22,283 effective boxes.
- `IP087000986.xml` is parsed as duplicated identical documents and counted once. The zero-width box in `IP046000898.xml` is filtered while its valid peer remains.
- IP102 source labels remain auditable; detector labels are contiguous `1..96` (`0` is background); public outputs use frozen DLCPD-25 `class_id 0..202`.
- IP102 classes 50 and 51 share DLCPD-25 public class 97 and one detector label. Official test has no source class 61; no support or AP may be fabricated.

## Consumer rules

Algorithm code must instantiate `IP102DetectionDataset` with this directory's `train.txt` or `val.txt`, this directory's `class-map.json`, and the raw VOC root. Evaluation may use only the unchanged `test.txt` after training configuration is frozen. Do not rescan directories, reinterpret XML, infer IDs, change splits, or expose detector-private labels as public IDs.

Verify with:

```bash
python3 scripts/ip102/build_detection_contract.py --verify-only
python3 scripts/ip102/verify_detection_contract.py
sha256sum -c artifacts/data/ip102/checksums.sha256
```
