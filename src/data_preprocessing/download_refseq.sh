# note: this is temporary POC, final version should download multiple species
# TODO: decide what subset of RefSeq to include

wget https://ftp.ncbi.nlm.nih.gov/genomes/refseq/vertebrate_mammalian/Homo_sapiens/reference/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_rna.gbff.gz \
    -O data/refseq/homo_sapiens_rna.gbff.gz
