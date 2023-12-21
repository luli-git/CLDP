model_m
model_t

scores = torch.tensor([nb_dis, nb_drugs])
for each disease:
    text = get_disease_discription(disease)
    text_embedding = model_t(text)
    for each drug:
        smiles = get_smiles(drug)
        drug_embedding = model_m(smiles)
        scores[disease_idx][drug_idx] = dot_product(text_embedding, drug_embedding)

save(score)

top_k_dict = {}
for each disease:
    top_k_drugs = get_top_k_drugs(scores[disease_idx]) # list of (drug names, scores)
    top_k_dict[disease] =    .0

def get_top_k_drugs(disease):
    score = scores[disease_idx]
    for each drug:
        if (drug, disease) in training_set:
            score[drug_idx] = -inf
    return top_k(score)

