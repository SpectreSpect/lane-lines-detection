import random


class Pallete:
    def __init__(self):
        pass
    
    @staticmethod
    def generate_random_colors(num_colors, seed=42):
        random.seed(seed)
        colors = []

        for idx in range(num_colors):
            red = random.randint(0, 255)
            green = random.randint(0, 255)
            blue = random.randint(0, 255)

            color = [red, green, blue]

            colors.append(color)
        
        return colors


        