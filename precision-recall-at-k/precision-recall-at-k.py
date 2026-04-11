def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    top_k = recommended[:k]
    precision_k = len(set(top_k).intersection(relevant))/k
    recall_k = len(set(top_k).intersection(relevant))/len(relevant)
    return [precision_k,recall_k]
    # Write code here