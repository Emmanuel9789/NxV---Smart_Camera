def threat_score(weapon, loitering, aggression):
    score = 0
    
    if weapon:
        score += 0.6
        
    if loitering:
        score += 0.3
        
    if aggression:
        score += 0.4
        
    return score