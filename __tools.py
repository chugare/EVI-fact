

str = """rouge-1 f : 0.283107
rouge-1 p : 0.466776
rouge-1 r : 0.247308
rouge-2 f : 0.091257
rouge-2 p : 0.173276
rouge-2 r : 0.077695
rouge-l f : 0.145787
rouge-l p : 0.310457
rouge-l r : 0.143363"""
ls = str.split('\n')
name = []
res = []
for l in ls:
    k = l.split(' ')
    name.append(k[0]+'-'+k[1])
    res.append(k[-1])
print('\n'.join(name))
print('\n'.join(res))