# TeamML-mRNA-project

## Environment setup

TODO

## Data preprocessing

### Pretraining data

For pretraining we use full mRNA sequences (5' UTR + CDS + 3' UTR) obtained
from [NCBI RefSeq] database.

First, run
```
python src/data_preprocessing/download_refseq.py
```
to download mRNA data from RefSeq FTP server to local `data/refseq`.
Modify `REFSEQ_URLS` list in the download script to download subset of data.

Next, run `src/data_preprocessing/filter_refseq.py` to parse all GBFF files in
`data/refseq`, filter valid sequences, and save them into csv files. To adjust
filtering parameters, modify config in the `filter_refseq.py` script.

Output csv files have columns `id`, `utr5`, `cds`, `utr3`. They can be finally
used to initialize `MRNACsvDataset`.

[NCBI RefSeq]: https://www.ncbi.nlm.nih.gov/refseq/
