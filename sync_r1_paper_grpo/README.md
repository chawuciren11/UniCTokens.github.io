# Sync-R1 Paper GRPO Adaptation

This folder contains a cleaned Sync-R1 / GRPO adaptation prepared on top of the UniCTokens codebase.

## Included Files

- `train_grpo_paper.py`
- `grpo_paper.py`
- `ref_model.py`
- `glm_api.py`
- `utils.py`
- `clip_eval.py`
- `pdata.py`
- `requirements.txt`
- `configs/showo_demo.yaml`
- `configs/showo_demo_512x512.yaml`

## Notes

- This release keeps only the paper-aligned GRPO path and excludes older draft variants.
- The GRPO implementation records MaskGIT rollout trajectories and computes trajectory-level policy ratios.
- The files in this folder are provided as a standalone adaptation bundle, but they still rely on the main UniCTokens repository modules at the repo root, such as `models/`, `training/`, and `llava/`.
- Model paths and dataset paths are placeholders and should be filled in locally.
- LLM-based scoring credentials are read from environment variables:
  - `ZAI_API_KEY`
  - `VERTEXAI_PROJECT`
  - `VERTEXAI_LOCATION`
  - `GOOGLE_APPLICATION_CREDENTIALS`

## Launch

Run from the repository root with the repo root available on `PYTHONPATH`.

PowerShell example:

```powershell
$env:PYTHONPATH='.'
torchrun --nproc_per_node=3 sync_r1_paper_grpo/train_grpo_paper.py `
  --num_gpus 3 `
  --config_file sync_r1_paper_grpo/configs/showo_demo_512x512.yaml `
  --data_root path/to/unictokens_data `
  --pre_trained_ckpt_name path/to/second_stage_checkpoint_dir `
  --concept adrien_brody `
  --save_dir ./tmp_result_accelerate/ `
  --epoch_to_load 15 `
  --batch_num 10 `
  --batch_size 1 `
  --num_gen 9 `
  --llm glm `
  --accelerate True `
  --semantic True
```
