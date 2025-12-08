from collections import Counter
import math

def calculate_precision(candidate, reference, n):
    """Calcula precisão n-gram."""
    candidate_ngrams = []
    for i in range(len(candidate) - n + 1):
        candidate_ngrams.append(tuple(candidate[i:i+n]))
        
    reference_ngrams = []
    for i in range(len(reference) - n + 1):
        reference_ngrams.append(tuple(reference[i:i+n]))
        
    candidate_counts = Counter(candidate_ngrams)
    reference_counts = Counter(reference_ngrams)
    
    overlap = 0
    for ngram, count in candidate_counts.items():
        overlap += min(count, reference_counts[ngram])
        
    total = len(candidate_ngrams)
    if total == 0:
        return 0
        
    return overlap / total

def bleu_score(candidate_text, reference_text):
    """
    Implementação simplificada do BLEU score.
    """
    candidate = candidate_text.lower().split()
    reference = reference_text.lower().split()
    
    precisions = []
    for n in range(1, 5):
        p = calculate_precision(candidate, reference, n)
        precisions.append(p)
        
    if min(precisions) == 0:
        return 0
        
    log_sum = sum([math.log(p) for p in precisions])
    geo_mean = math.exp(log_sum / 4)
    
    # Brevity penalty
    c = len(candidate)
    r = len(reference)
    bp = 1 if c > r else math.exp(1 - r/c)
    
    return bp * geo_mean

if __name__ == "__main__":
    ref = "O mosquito da dengue se reproduz em água parada"
    cand = "O mosquito reproduz em água limpa e parada"
    
    print(f"Reference: {ref}")
    print(f"Candidate: {cand}")
    print(f"BLEU: {bleu_score(cand, ref):.4f}")
