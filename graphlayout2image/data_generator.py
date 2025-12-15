from random import random
import torch 
import numpy as np 
from PIL import Image, ImageDraw
import random 


class SimpleShapeScene:
    """Generate 32x32 images with 2-3 colored shapes"""
    SHAPES = ['circle', 'square', 'triangle']
    COLORS = ['red', 'blue', 'green', 'yellow']

    def generate_scene(self ):
        img = Image.new("RGB", (32, 32), "white")
        draw = ImageDraw.Draw(img)

        obj1 = {
            "shape": random.choice(self.SHAPES),
            "color": random.choice(self.COLORS),
            "x": random.randint(5, 12),
            "y": random.randint(5, 27),
        }

        relation = random.choice(['left_of', 'right_of', 'above', 'below'])
        obj2 = self.generate_related_object(obj1, relation)

        self.draw_shape(draw, obj1)
        self.draw_shape(draw, obj2)

        graph = {
            "nodes": [
                {'shape': obj1['shape'], 'color': obj1['color']},
                {'shape': obj2['shape'], 'color': obj2['color']}
            ],
            "edges": [{'relation': relation}]
        }

        return np.array(img), graph 

    def draw_shape(self, draw, obj):
        x, y = obj['x'], obj['y']
        color = obj['color']
        
        if obj['shape'] == 'circle':
            draw.ellipse([x, y, x + 6, y + 6], fill=color)
        elif obj['shape'] == 'square':
            draw.rectangle([x, y, x + 6, y + 6], fill=color)
        elif obj['shape'] == 'triangle':
            draw.polygon([(x + 3, y), (x, y + 6), (x + 6, y + 6)], fill=color)
    
    def generate_related_object(self, obj1, relation):
        obj2 = {
            'shape': random.choice(self.SHAPES),
            'color': random.choice([c for c in self.COLORS if c != obj1['color']])
        }
        
        if relation == 'right_of':
            obj2['x'] = obj1['x'] + 10
            obj2['y'] = obj1['y'] + random.randint(-3, 3)
        elif relation == 'left_of':
            obj2['x'] = obj1['x'] - 10
            obj2['y'] = obj1['y'] + random.randint(-3, 3)
        elif relation == 'above':
            obj2['x'] = obj1['x'] + random.randint(-3, 3)
            obj2['y'] = obj1['y'] - 10
        else:  
            obj2['x'] = obj1['x'] + random.randint(-3, 3)
            obj2['y'] = obj1['y'] + 10
        
        obj2['x'] = max(2, min(24, obj2['x']))
        obj2['y'] = max(2, min(24, obj2['y']))
        
        return obj2


def main():
    generator = SimpleShapeScene()
    dataset = [generator.generate_scene() for _ in range(2000)]
    torch.save(dataset, 'simple_scenes.pt')

if __name__ == "__main__":
    main()
