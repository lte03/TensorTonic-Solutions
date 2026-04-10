def calc_vector_value(v):
    return math.sqrt(sum([a**2 for a in v]))

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    cos = sum([a*b for a,b in zip(x1,x2)])/(calc_vector_value(x1) * calc_vector_value(x2))
    if label == 1:
        loss = 1 - cos
    else:
        loss = max(0,cos - margin)
    return loss