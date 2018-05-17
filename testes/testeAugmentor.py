import Augmentor

pathImagens = "/home/geovane/Imagens/teste/imagens/augmentor"

p = Augmentor.Pipeline(pathImagens)

p.flip_top_bottom(probability=0.7)
p.flip_left_right(probability=0.5)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)

p.sample(20)