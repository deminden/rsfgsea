# Extracts gene ID mappings from the bundled R data file and exports JSON lookup tables used by local data-prep scripts.
# Load the RDA file
load("data/Gene_to_id_mapping.rda")

library(jsonlite)

# Create a named list for JSON
mapping_list <- setNames(as.list(gene_mapping$GeneName), gene_mapping$GeneID)

write_json(mapping_list, "data/gene_to_id_mapping_full.json", auto_unbox = TRUE, pretty = TRUE)
print(paste("Saved", length(mapping_list), "gene mappings"))

# Also create reverse mapping (Symbol -> Ensembl)
reverse_mapping <- setNames(as.list(gene_mapping$GeneID), gene_mapping$GeneName)
write_json(reverse_mapping, "data/symbol_to_ensembl.json", auto_unbox = TRUE, pretty = TRUE)
print("Saved reverse mapping")
