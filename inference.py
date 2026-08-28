import argparse
import os
import torch
from torch.utils.data import DataLoader
from models.xacle_baseline_model import XACLEBaselineModel
from datasets.xacle_baseline_dataset import get_infdataset
import utils.utils as utils
from utils.checkpoint import load_trainable_checkpoint
from utils.config import load_config
from utils.runtime import seed_everything
from tqdm import tqdm
import csv

def inference():
    # -------- initial setup --------
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_directory")
    parser.add_argument("dataset", nargs="?", choices=["validation", "test"], default="validation")
    args = parser.parse_args()

    chkpt_dir = args.checkpoint_directory
    if not os.path.isdir(chkpt_dir):
        chkpt_dir = os.path.join("./chkpt", chkpt_dir)
    if not os.path.isdir(chkpt_dir):
        parser.error(f"checkpoint directory does not exist: {chkpt_dir}")
    chkpt_path = os.path.join(chkpt_dir, "best_model.pt")
    cfg_path   = os.path.join(chkpt_dir, "config.json")
    if not os.path.isfile(chkpt_path):
        parser.error(f"checkpoint does not exist: {chkpt_path}")
    if not os.path.isfile(cfg_path):
        parser.error(f"run config does not exist: {cfg_path}")
    dataset_key = args.dataset
    cfg = load_config(cfg_path)
    seed_everything(cfg["seed"])
    dataset_label   = f"{dataset_key}_list"
    dataset_list    = cfg[dataset_label]
    dataset_wav_dir = os.path.join(cfg["wav_dir"], dataset_key)
    print("Perform inference on the following dataset with following checkpoint.")
    print(f"\tchkpt:        {chkpt_path}")
    print(f"\tdata list:    {dataset_list}")
    print(f"\twav dir:      {dataset_wav_dir}") 
    device = torch.device(cfg["device"])
    # -------------------------------

    # -------- dataset / dataloader --------
    test_ds   = get_infdataset(
        txt_file_path=dataset_list,
        wav_dir=dataset_wav_dir,
        max_sec=cfg["max_len"],
        sr=cfg["m2d_clap"]["sample_rate"]
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.get("inference_batch_size", cfg.get("val_batch_size", 1)),
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=test_ds.collate_fn
    )
    # -------------------------------------------------

    # -------- model --------
    model = XACLEBaselineModel(cfg, device).to(device)
    load_trainable_checkpoint(model, chkpt_path, map_location=device)
    model.eval()
    # ------------------------------------

    # -------- run inference --------
    rows = []
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(test_loader)):
            if batch_index >= cfg.get("max_inference_batches", len(test_loader)):
                break
            batch = utils.move_to_device(batch, device)
            pred = model(batch).detach().cpu()
            pred_mos = pred * 5.0 + 5.0
            rows.extend(
                {
                    "wav_file_name": os.path.basename(wav_path),
                    "pred_score": round(score.item(), 2),
                }
                for wav_path, score in zip(batch["wav_paths"], pred_mos)
            )
    # -------------------------------

    # -------- write results --------
    result_path = os.path.join(chkpt_dir, f"inference_result_for_{dataset_key}.csv")
    print(f"Inference has completed. Results will be written to the following file: \n\t{result_path}")
    with open(result_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wav_file_name", "pred_score"])
        writer.writeheader()
        writer.writerows(rows)
    # -------------------------------

    return

if __name__ == "__main__":
    inference()
