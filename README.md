# TeamML-mRNA-project

## Environment setup

TODO

## Data preprocessing

### Pretraining data

For pretraining we use full mRNA sequences (5' UTR + CDS + 3' UTR) obtained
from [NCBI RefSeq] database.

`src/data_preprocessing/download_refseq.py` script was used to download RNA
sequences from RefSeq FTP server.

`src/data_preprocessing/generate_refseq_metadata.py` was used to extract
sequences metadata from raw GBFF files.

Finally, `src/data_preprocessing/filter_refseq.py` was used to parse sequences
and filter them into the final dataset.

### Usage

Pretraining dataset is available under `data/pretraining/pretraining_refseq.csv`
This csv file has columns `id`, `utr5`, `cds`, `utr3`. Example usage below:

```
from src.transformer_training.pretrain import MRNACsvDataset

data_path = "data/pretraining/pretraining_refseq.csv"
dataset = MRNACsvDataset(data_path)
```

[NCBI RefSeq]: https://www.ncbi.nlm.nih.gov/refseq/
