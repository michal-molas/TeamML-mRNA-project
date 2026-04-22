import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "RiboNN"))
from RiboNN.src.model import RiboNN
from models import MRNACsvDataset, MRNATransformer

RIBONN_MAX_TX_LEN = 1_381 + 11_937  # 13318

# len_after_conv: sequence length after all 10 conv+pool layers for a 13318-length input.
# Derivation:
#   1. initial_conv(k=5,p=0): 13318 -> 13314
#   2. 10 × (conv(k=5,p=0) + maxpool(2,2)): 13314 -> 13310 -> 6655 -> 6651 -> 3325 -> 3321 -> 1660 -> 1656 -> 828 -> 824 -> 412 -> 408 -> 204 -> 200 -> 100 -> 96 -> 48 -> 44 -> 22 -> 18 -> 9
RIBONN_LEN_AFTER_CONV = 9

RIBONN_CONFIG = dict(
    with_NAs=False, # conf.yml
    split_utr5_cds_utr3_channels=False, # conf.yml
    label_codons=True, # conf.yml
    label_utr5=False, # conf.yml
    label_utr3=False, # conf.yml
    label_splice_sites=False, # conf.yml
    label_up_probs=False, # conf.yml
    filters=64, # conf.yml
    conv_stride=1, # conf.yml
    conv_padding=0, # conf.yml
    ln_epsilon=0.007, # conf.yml
    dropout=0.3, # conf.yml
    residual=False, # conf.yml
    activation = "relu", # Can be also silu/leakyrelu
    kernel_size=5, # conf.yml
    num_conv_layers=10, # conf.yml
    len_after_conv=RIBONN_LEN_AFTER_CONV,
    num_targets=78,  # 78 for human, 68 for mouse
    max_shift=0, # conf.yml
    symmetric_shift=True, # conf.yml
)


def load_ribonn(weights_path, device):
    """Load frozen RiboNN weights from the submodule. Returns (model, RIBONN_MAX_TX_LEN)."""

    config = dict(RIBONN_CONFIG)
    model = RiboNN(**config)

    state_dict = torch.load(weights_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"[ribonn] loaded {weights_path}  missing={len(missing)}  unexpected={len(unexpected)}"
    )

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, RIBONN_MAX_TX_LEN


def build_ribonn_input(
    lm_logits,
    input_ids,
    utr5_lens,
    cds_lens,
    utr3_lens,
    ribonn_max_len,
    label_codons=True,
):
    """
    Convert LM logits into a RiboNN-compatible tensor via Gumbel-softmax.

    Vocab order A=0,U/T=1,C=2,G=3 matches RiboNN channels.
    RiboNN input layout: [ utr5 | cds | utr3 | padding ]  (zero-padded to ribonn_max_len)
    UTR5 is stored reversed in the LM; we flip it back here.

    Returns: (N, num_channels, ribonn_max_len)  — channels: A,U/T,C,G [+ codon_label]
    """
    batch_size = lm_logits.size(0)
    device = lm_logits.device
    num_channels = 5 if label_codons else 4
    out = torch.zeros(batch_size, num_channels, ribonn_max_len, device=device)

    # Nucleotide channels 0-3 already match RiboNN, but we have to drop special-token logits
    soft_nt = F.gumbel_softmax(lm_logits)[:, :, :4]

    for i in range(batch_size):
        utr5_len = int(utr5_lens[i].item())
        cds_len  = int(cds_lens[i].item())
        utr3_len = int(utr3_lens[i].item())

        # UTR5
        utr5_start = 2 + cds_len + 1 # after <BOS>, <CDS>, CDS tokens and <UTR5>
        utr5_end = utr5_start + utr5_len

        utr5_out_start = 0
        utr5_out_end = utr5_len
        out[i, :4, utr5_out_start:utr5_out_end] = soft_nt[i, utr5_start:utr5_end].flip(0).T

        # CDS (Here we use one-hot encoding instead of Gumbel-softmax)
        cds_start = 2 # after <BOS>, <CDS>
        cds_end = cds_start + cds_len

        cds_tokens = input_ids[i, cds_start:cds_end]
        cds_out_start = utr5_len
        cds_out_end = utr5_len + cds_len
        out[i, :4, cds_out_start:cds_out_end] = F.one_hot(cds_tokens, 4).float().T

        # UTR3
        utr3_start = 2 + cds_len + 1 + utr5_len + 1 # after <BOS>, <CDS>, CDS tokens, <UTR5>, UTR5 tokens and <UTR3> token 
        utr3_end = utr3_start + utr3_len

        utr3_out_start = utr5_len + cds_len
        utr3_out_end = utr5_len + cds_len + utr3_len
        out[i, :4, utr3_out_start:utr3_out_end] = soft_nt[i, utr3_start:utr3_end].T

        # Codon-label channel
        if label_codons:
            # Label every first nucleotide of a codon (every 3rd position) in CDS (UTRs should not be labeled)
            for codon_pos in range(cds_out_start, cds_out_end, 3):
                if codon_pos < cds_out_end:
                    out[i, 4, codon_pos] = 1.0

    return out  # (N, num_channels, ribonn_max_len)


def load_pretrained_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded {checkpoint_path}  missing={len(missing)}  unexpected={len(unexpected)}"
    )


def train(args, device):

    dataset = MRNACsvDataset(
        csv_path=args.csv_path,
        max_utr5_len=args.max_utr5_len,
        max_cds_len=args.max_cds_len,
        max_utr3_len=args.max_utr3_len,
    )

    dataset_size = len(dataset)

    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"Train dataset length: {len(train_dataset)}", file=sys.stderr)
    print(f"Validation dataset length: {len(val_dataset)}", file=sys.stderr)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MRNATransformer(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        nhead=args.n_heads,
        num_layers=args.n_layers,
        max_len=dataset.max_len,
    ).to(device)

    load_pretrained_weights(model, args.pretrained_path, device)

    # ribonn_model, ribonn_max_len = load_ribonn(args.ribonn_weights, device)
    _, ribonn_max_len = load_ribonn(args.ribonn_weights, device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    global_step = 0
    best_val_lm = float("inf")
    for epoch in range(args.epochs):
        model.train()
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]
            loss_mask = batch["loss_mask"]
            padding_mask = batch["padding_mask"]
            utr5_lens = batch["utr5_len"]
            cds_lens = batch["cds_len"]
            utr3_lens = batch["utr3_len"]
            te_label = batch["te_label"]

            optimizer.zero_grad()
            lm_logits = model(input_ids, padding_mask=padding_mask)

            # LM loss
            outputs_flat = lm_logits.view(-1, lm_logits.size(-1))
            targets_flat = target_ids.view(-1)
            
            lm_loss = F.cross_entropy(outputs_flat, targets_flat, reduction='none')

            lm_loss = (lm_loss * loss_mask.view(-1)).sum() / (loss_mask.sum() + 1e-8)

            # RiboNN loss
            ribonn_loss = torch.tensor(0.0, device=device)
            ribonn_input = build_ribonn_input(
                lm_logits,
                input_ids,
                utr5_lens,
                cds_lens,
                utr3_lens,
                ribonn_max_len,
                label_codons=RIBONN_CONFIG["label_codons"],
            )

            ## RiboNN.src.predict.predict_using_nested_cross_validation_models() ##
            run_df = pd.read_csv(args.ribonn_weights_folder + '/runs.csv') 
            all_prediction_dfs = []
            for test_fold in np.sort(run_df["params.test_fold"].unique()):
                test_fold_str = str(test_fold)
                sub_run_df = run_df.query(
                    "`params.test_fold` == @test_fold_str or `params.test_fold` == @test_fold"
                ).reset_index(drop=True)

                # prediction_df = predict_using_models_trained_in_one_fold(
                #     sub_run_df, config, dm, top_k_models_to_use
                # )

                ## RiboNN.src.predict.predict_using_models_trained_in_one_fold() ##
                top_k_models_to_use = 1
                training_data_columns = "TE_108T,TE_12T,TE_A2780,TE_A549,TE_BJ,TE_BRx.142,TE_C643,TE_CRL.1634,TE_Calu.3,TE_Cybrid_Cells,TE_H1.hESC,TE_H1933,TE_H9.hESC,TE_HAP.1,TE_HCC_tumor,TE_HCC_adjancent_normal,TE_HCT116,TE_HEK293,TE_HEK293T,TE_HMECs,TE_HSB2,TE_HSPCs,TE_HeLa,TE_HeLa_S3,TE_HepG2,TE_Huh.7.5,TE_Huh7,TE_K562,TE_Kidney_normal_tissue,TE_LCL,TE_LuCaP.PDX,TE_MCF10A,TE_MCF10A.ER.Src,TE_MCF7,TE_MD55A3,TE_MDA.MB.231,TE_MM1.S,TE_MOLM.13,TE_Molt.3,TE_Mutu,TE_OSCC,TE_PANC1,TE_PATU.8902,TE_PC3,TE_PC9,TE_Primary_CD4._T.cells,TE_Primary_human_bronchial_epithelial_cells,TE_RD.CCL.136,TE_RPE.1,TE_SH.SY5Y,TE_SUM159PT,TE_SW480TetOnAPC,TE_T47D,TE_THP.1,TE_U.251,TE_U.343,TE_U2392,TE_U2OS,TE_Vero_6,TE_WI38,TE_WM902B,TE_WTC.11,TE_ZR75.1,TE_cardiac_fibroblasts,TE_ccRCC,TE_early_neurons,TE_fibroblast,TE_hESC,TE_human_brain_tumor,TE_iPSC.differentiated_dopamine_neurons,TE_megakaryocytes,TE_muscle_tissue,TE_neuronal_precursor_cells,TE_neurons,TE_normal_brain_tissue,TE_normal_prostate,TE_primary_macrophages,TE_skeletal_muscle"
                predicted_columns = training_data_columns.replace("TE_", "predicted_TE_").split(",")

                sub_run_df = sub_run_df.sort_values("metrics.val_r2", ascending=False).head(top_k_models_to_use)

                predictions = []
                # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                for run_id in sub_run_df.run_id:
                    # Create a new model
                    local_state_dict_path = f"{args.ribonn_weights_folder}/{run_id}/state_dict.pth"
                    ribonn_model, _ = load_ribonn(local_state_dict_path, device)
                    ribonn_model.to(device)
                    ribonn_model.eval()
                    predictions.append(ribonn_model(ribonn_input))

                mean_prediction = torch.stack(predictions, axis=-1).mean(axis=-1)
                print(mean_prediction.shape, file=sys.stderr)

                fold_df = pd.DataFrame(mean_prediction.detach().numpy(), columns=predicted_columns)
                print(fold_df, file=sys.stderr)
                print(fold_df.shape, file=sys.stderr)

                sys.exit(0)

                # df = pd.concat([dm.df, df], axis=1)
                # return df
                ##

                fold_df["fold"] = int(test_fold)
                all_prediction_dfs.append(fold_df)

            all_predictions = pd.concat(all_prediction_dfs, axis=0, ignore_index=True)
            ##

            # te_pred = ribonn_model(ribonn_input)

            # TODO: How to compare the TE to the label? 
            #       Can our model output results better than the data (then maybe relu)?
            #       Maybe we should just try to maximize the TE and ignore the label?
            ribonn_loss = F.mse_loss(te_pred.mean(dim=-1), te_label)

            loss = lm_loss + args.lambda_ribonn * ribonn_loss
            loss.backward()
            optimizer.step()

            global_step += 1

            if args.wandb and step % 100 == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lm_loss": lm_loss.item(),
                        "train/ribonn_loss": ribonn_loss.item(),
                    },
                    step=global_step,
                )

        model.eval()
        val_loss_sum = 0
        val_te_sum = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                input_ids = batch["input_ids"]
                target_ids = batch["target_ids"]
                loss_mask = batch["loss_mask"]
                padding_mask = batch["padding_mask"]
                utr5_lens = batch["utr5_len"]
                cds_lens = batch["cds_len"]
                utr3_lens = batch["utr3_len"]

                lm_logits = model(input_ids, padding_mask=padding_mask)
                lm = F.cross_entropy(
                    lm_logits.view(-1, lm_logits.size(-1)),
                    target_ids.view(-1),
                    reduction="none",
                )
                lm = (lm * loss_mask.view(-1)).sum() / (loss_mask.sum() + 1e-8)
                val_loss_sum += lm.item()

                discrete_logits = F.one_hot(lm_logits.argmax(-1), num_classes=lm_logits.size(-1)).float()
                ribonn_input = build_ribonn_input(
                    discrete_logits,
                    input_ids,
                    utr5_lens,
                    cds_lens,
                    utr3_lens,
                    ribonn_max_len,
                    label_codons=RIBONN_CONFIG["label_codons"],
                )
                val_te_sum += ribonn_model(ribonn_input).mean().item()

                val_n += 1

        val_lm = val_loss_sum / val_n
        val_te = val_te_sum / val_n

        print(
            f"Epoch {epoch} | train_lm={lm_loss.item():.4f} "
            f"train_ribonn={ribonn_loss.item():.4f}  val_lm={val_lm:.4f}  val_te={val_te:.4f}"
        )

        if args.wandb:
            wandb.log({"val/lm_loss": val_lm, "val/te": val_te, "epoch": epoch}, step=global_step)

        if val_lm < best_val_lm:
            best_val_lm = val_lm
            torch.save(model.state_dict(), args.output_path)
            print(f"[checkpoint] saved → {args.output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path", type=str, default="../../data/finetuning/ribonn/dataset.csv"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to pretrain.py checkpoint",
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--ribonn_weights",
        type=str,
        default=None,
        help="Path to a RiboNN state_dict.pth weight file",
    )
    parser.add_argument(
        "--ribonn_weights_folder",
        type=str,
        default=None,
        help="Path to a RiboNN state_dict.pth weight file",
    )
    parser.add_argument(
        "--lambda_ribonn",
        type=float,
        default=1.0,
        help="Weight of RiboNN loss relative to LM loss",
    )

    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--max_utr5_len", type=int, default=200)
    parser.add_argument("--max_cds_len", type=int, default=500)
    parser.add_argument("--max_utr3_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_dotenv()

    if args.wandb:
        wandb.init(
            project="teamml-ribonn-finetune",
            config=vars(args),
            dir='../../logs',
        )

    train(args, device)

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
