import matplotlib.pyplot as plt

#     Liste des scores des tableaux : [0.9681514109089986, 0.0, 0.9678268311817987, 0.6751022878197164, 0.9281086366105378, 0.9730270711221222, 0.38827020566294357, 0.8732378880572986, 0.9800637997809858, 0.9592386314951955, 0.011453270320725205, 0.9661309832382927, 0.9401947238927933, 0.3845607137456525, 0.0, 0.8432739354826355, 0.8177145363160678, 0.6578140226885644, 0.9052559640636134, 0.9731000633312223, 0.9407248513942021, 0.9859830975667362, 0.5925052886068298, 0.9892725726728524, 0.7950229055189859, 0.9845840201821597, 0.8993850368602202, 0.9266821424860865]#     Liste des scores des lignes : [0.0, 0.40218402469449027, 0.7004085810957591, 0.39558188332524796, 0.7305310253599298, 0.6168369407204006, 0.06071995322809655, 0.15522960519777715, 0.6515562105710971, 0.6258985870493293, 0.0, 0.6762815884644153, 0.37087948394609155, 0.5890850264071942, 0.5298203053420829, 0.5767110592561687, 0.4973222288083821, 0.0, 0.6190585330452278, 0.719093565745771, 0.06163270961636542, 0.415239277952721, 0.3493233155349164, 0.677510401084182, 0.6492080599971507, 0.6107408693551553, 0.42052104898379455, 0.6989801571986993]
#     Liste des scores des lignes : [0.0, 0.0, 0.6601725229926584, 0.30449967760754093, 0.7181288768175893, 0.5355054402292049, 0.0, 0.16212676631949174, 0.656009771054243, 0.6297964912280701, 0.0, 0.6303265154281108, 0.40604658566151947, 0.07157105408820795, 0.0, 0.5402646570351983, 0.5391612369315154, 0.3644477602156708, 0.6236423676473699, 0.7765179118900022, 0.06625188771892347, 0.3569261880687563, 0.5120878989680365, 0.6273574471168403, 0.15464384486458357, 0.6280779713236165, 0.3566129005629757, 0.7409912502316689]

liste = \
[0.0, 0.0, 0.6601725229926584, 0.30449967760754093, 0.7181288768175893, 0.5355054402292049, 0.0, 0.16212676631949174, 0.656009771054243, 0.6297964912280701, 0.0, 0.6303265154281108, 0.40604658566151947, 0.07157105408820795, 0.0, 0.5402646570351983, 0.5391612369315154, 0.3644477602156708, 0.6236423676473699, 0.7765179118900022, 0.06625188771892347, 0.3569261880687563, 0.5120878989680365, 0.6273574471168403, 0.15464384486458357, 0.6280779713236165, 0.3566129005629757, 0.7409912502316689]

liste = [i * 100 for i in liste]
tot = []
for i in range(101):
    cpt = 0
    for l in liste:
        if l > i:
            cpt += 1

    tot.append(cpt / len(liste) * 100)
    print(f"{i} : {cpt / len(liste) * 100} %")

plt.plot(tot)
# add axes labels
plt.xlabel('Seuil de validation (en %)')
plt.ylabel("% d'images valides")
# plt.title(sum(tot))
plt.savefig('plot.png')
plt.show()

print(sum(tot))
