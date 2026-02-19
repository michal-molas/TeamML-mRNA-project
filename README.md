# TeamML-mRNA-project

## Data preprocessing

### Pretraining data

First, we download transcripts from RefSeq database to pretrain on full
5'UTR + CDS + 3'UTR sequences. To do this, run `src/data_preprocessing/download_refseq.sh`.

Next, run `src/data_preprocessing/preprocess_refseq.py` to parse all
GBFF files in `data/refseq` and merge them into single csv with `utr5`,
`cds`, `utr3` strings, and metadata: RefSeq accsesion ID `acc` and
source file `source`.
