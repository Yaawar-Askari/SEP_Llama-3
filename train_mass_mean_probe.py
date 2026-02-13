# Conceptual logic
def get_mass_mean_direction(X, y):
    # X: (N, Dim)
    # y: (N,)
    
    # 1. Separate classes
    class_0 = X[y == 0] # Confident (Truth)
    class_1 = X[y == 1] # Hallucinated
    
    # 2. Compute Means
    mean_0 = torch.mean(class_0, dim=0)
    mean_1 = torch.mean(class_1, dim=0)
    
    # 3. The "Hallucination Direction"
    # This vector points from Truth -> Hallucination
    direction = mean_1 - mean_0
    
    return direction

# To predict:
# Project new sample x onto direction: dot(x, direction)
# If > threshold, it's hallucinated.