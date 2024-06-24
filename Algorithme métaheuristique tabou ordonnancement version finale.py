from copy import deepcopy
import numpy as np
import random as rd

## Note au lecteur: Utilisez trace_Sol(s) pour tracer l'emploi du temps correspondant à la solution s
#exemple: trace_Sol(c)

## Déclaration de variables globales et mémorisation de solutions remarquables

n = 10 #nombre de patients
tableDeComp = [[[1, 1], [1, 2], [1, 3], [1, 2, 2], [4, 5, 5, 6]], #Table de compétences
    [[2, 3], [2, 3], [2]],
    [[3, 3], [3]],
    [[4, 4], [5, 6], [6, 6], [4, 4], [1, 2]],
    [[2, 2], [5], [5, 6], [4, 5], [3]],
    [[1], [4], [6]],
    [[6, 6], [1], [5, 6], [3]],
    [[3, 5], [2, 5], [3, 6], [6]],
    [[5], [4], [1]],
    [[4], [4, 5], [1, 2], [4]]]

# solution optimale de l'algo Tabou
c = [[(1, 6),(1, 1),(1, 1),(2, 1),(3, 9),(2, 7),(3, 1),(5, 4),(4, 1),(-1, -1),(3, 10),(-1, -1)],
 [(1, 2),(1, 5),(1, 5),(2, 1),(2, 8),(2, 2),(3, 2),(5, 4),(4, 1),(4, 1),(3, 10),(-1, -1)],
 [(1, 2),(1, 8),(1, 3),(1, 3),(2, 3),(2, 2),(3, 1),(4, 7),(3, 8),(5, 5),(-1, -1),(-1, -1)],
 [(1, 4),(1, 4),(2, 9),(2, 6),(1, 10),(4, 4),(4, 4),(2, 10),(4, 5),(-1, -1),(5, 1),(4, 10)],[(1, 9),(1, 8),(2, 4),(2, 5),(2, 8),(3, 5),(3, 7),(2, 10),(4, 5),(-1, -1),(5, 1),(5, 1)],
 [(1, 7),(1, 7),(2, 4),(3, 4),(3, 4),(3, 5),(3, 7),(3, 6),(3, 8),(4, 8),(5, 1),(-1, -1)]]

# solution de l'algo de Liste
s0Liste = [
[(1, 1), (1, 1), (1, 6), (2, 1), (2, 7), (3, 9), (3, 1), (4, 1), (-1, -1), (3, 10), (5, 4), (-1, -1)],
[(1, 2), (1, 5), (1, 5), (2, 1), (2, 2), (2, 8), (3, 2), (4, 1), (4, 1), (3, 10), (5, 4), (-1, -1)],
[(1, 2), (1, 3), (1, 3), (1, 8), (2, 2), (2, 3), (3, 1), (3, 8), (4, 7), (-1, -1), (5, 5), (-1, -1)],
[(1, 4), (1, 4), (1, 10), (2, 6), (2, 9), (4, 4), (4, 4), (2, 10), (-1, -1), (4, 5), (4, 10), (5, 1)],
[(1, 9), (-1, -1), (2, 4), (1, 8), (2, 5), (2, 8), (3, 7), (2, 10), (3, 5), (4, 5), (-1, -1), (5, 1), (5, 1)],
[(1, 7), (1, 7), (2, 4), (3, 4), (3, 4), (3, 6), (3, 7), (3, 8), (3, 5), (4, 8), (-1, -1), (5, 1)]]

# solution donnée en exemple
s1 = [
[(1, 1), (1, 1), (-1, -1), (1, 6), (-1, -1), (2, 7), (3, 1), (-1, -1), (-1, -1), (4, 1), (3, 9),(-1, -1), (3, 10), (5, 4)],
[(1, 5), (1, 5), (1, 2), (2, 1), (-1, -1), (2, 8), (2, 2), (-1, -1), (3, 2), (-1, -1),(-1, -1), (4, 1), (4, 1), (3, 10), (5, 4)],
[(1, 8), (-1, -1), (1, 2), (-1, -1), (1, 3), (1, 3), (2, 2), (3, 1), (-1, -1), (2, 3), (3, 8), (-1, -1), (-1, -1), (5, 5), (4, 7)],
[(1, 4), (1, 4), (-1, -1), (1, 10), (-1, -1), (2, 9), (-1, -1), (2, 10), (2, 6), (-1, -1), (4, 4), (4, 4), (4, 5), (5, 1), (4, 10)],
[(1, 8), (-1, -1), (1, 9), (2, 5), (2, 4), (2, 8), (3, 5), (2, 10), (-1, -1), (3, 7), (-1, -1), (-1, -1), (4, 5), (5, 1), (5, 1)],
[(1, 7), (1, 7), (-1, -1), (-1, -1), (2, 4), (-1, -1), (3, 5), (3, 4), (3, 4), (3, 7), (3, 8), (3, 6), (-1, -1), (4, 8), (5, 1)]
]

## Création de solutions peu intéressantes pour évaluer l'efficacité de l'algorithme Tabou

def random_empty_insert(s): #IN PLACE : insère un temps d'inactivité pour une compétence
    machine = rd.randrange(len(s))
    cycle = rd.randrange(len(s[machine]))
    s[machine].insert(cycle,(-1,-1))

def create_filled_empty_sol(s,n): #Crée une solution à partir d'une solution s en rajoutant n temps d'inactivités permettant d'obtenir un emploi du temps compatible
    newS= deepcopy(s)
    for i in range(n):
        newSbis = deepcopy(newS)
        random_empty_insert(newSbis)
        if admissible(newSbis):
            newS = newSbis
    return newS




## Fonctions d'évaluation

def Cmax(s): # max (dernier cycle pour chaque opération)
    m = len(s)
    return max([len(s[k]) for k in range(m)]) # Cmax : maskespan


def f(s):  #Calcule le nombre de cases vides (1 case vide = 1 unité de durée d'inactivité d'une compétence) qui ne sont pas situées à une extremité droite
    SommeCasesVides = 0
    for k in range(len(s)):
        sommePotentielle = 0
        for l in range (len(s[k])):
            if s[k][l] != (-1,-1):
                SommeCasesVides += sommePotentielle
                sommePotentielle = 0
            else:
                sommePotentielle += 1
    return SommeCasesVides

def fbis(s): # Fonction d'évaluation retenue : Prend également en compte la position des cases vides (valeur plus faible pour une case vide à droite), en attribuant + d'importance malgré tout au nombre de cases vides
    SommeCasesVides = 0
    for k in range(len(s)): #on parcourt toute les cases de la solution
        sommePotentielle = 0
        for l in range (len(s[k])):
            if s[k][l] != (-1,-1): #on ne prend pas en compte les cases vides situées aux extrémités droites, car la compétence n'attend alors plus personne
                SommeCasesVides += sommePotentielle
                sommePotentielle = 0
            else:
                sommePotentielle += len(s[k])*len(s) + (len(s[k])-(l+1)) #-> On donne une grande priorité à l'absence de cases vides et une moins grane priorité à leur position
    return SommeCasesVides

## Fonctions diverses

def admissible(s): # retourne True si s est une solution admissible (ie les opérations sont effectuées dans l'ordre), False sinon
#n est le nombre de patients
    m = len(s)
    op = [[] for i in range(n)]
    cmax = Cmax(s)

    for temps in range(cmax): #on parcourt de haut en bas puis de gauche à droite
        #print('l=',l)
        for tache in range(m):
            #print('k=', k)
            if temps < len(s[tache]):
                i = s[tache][temps][1]
                j = s[tache][temps][0]
                if i != -1 and j not in op[i-1]:
                    op[i-1].append(j) # on ajoute les opérations pour chaque compétence k par ordre d'apparition dans le planning s
                    #print(i, j)
                    if j > len(op[i-1]) or op[i-1][j-1] != j: # si la jème opération du planning s pour le patient i n'est pas l'opération j, i.e., si le critère de précédence des opérations pour un patient donné n'est pas vérifié:
                        return False

    return True


## Affichage d'une solution


import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np

def trace_Sol(s, n=10): #On trace la représentation graphique de la solution s

    plt.clf()
    fig, gnt = plt.subplots()
    gnt.set_xlabel('cycle')
    gnt.set_ylabel('skill')
    gnt.set_yticks(np.arange(1, 7))

    loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    gnt.xaxis.set_major_locator(loc)


    def get_cmap(n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)

    cmap = get_cmap(n+1)

    legends = [0 for i in range(n)]

    for i in range(len(s)):
        #print('i=', i)
        for j in range(len(s[i])):
            if s[i][j] != (-1,-1):
                (operation,patient) = s[i][j]
                c = 0
                k = i
                #print('j=', j)

                #print('k=', k)
                if legends[patient-1] == 0:
                    gnt.broken_barh([(j, 1)], (k - 0.25, 0.5), facecolor = cmap(patient), edgecolor = 'k', label = 'Patient ' + str(patient))
                    #print(t[i][j][c])
                    legends[patient-1] = 1
                else:
                    gnt.broken_barh([(j, 1)], (k - 0.25, 0.5), facecolor = cmap(patient), edgecolor = 'k')
                    #print(t[i][j][c])
                gnt.text((2*j + 1)/2,  k, str(operation), ha = 'center', va = 'center', color = 'k')
                c += 1

    plt.legend()
    plt.show()

##Algo de liste utilisé dans le voisinnage

def ListeInitiale(tableDeComp): #On convertit une table de Compétence en une première liste de priorité en respectant le critère de priorité suivant: "prochaine op = i prioritaire par rapport à prochaine op = j si i<j"
    n = len(tableDeComp)
    J =[]
    for patient in range(n):
        J.append(len(tableDeComp[patient])) # nb d'opérations par patient
    # J = [5, 3, 2, 5, 5, 3, 4, 4, 3, 4]

    def second(elem):
        return elem[1]
    readyop = sorted([(i, j) for i in range(1, n+1) for j in range(1, J[i-1]+1)], key = second) # opérations prêtes à être exécutées
    return readyop

readyop = ListeInitiale(tableDeComp) #C'est la liste de priorité qui engendre la solution correspondant au premier algorithme de Liste

def AlgoListe(readyopbis, tableDeComp): #Il s'agit d'un algorithme de Liste applicable à toute liste de priorités readyopbis et qui prend en compte une table de compétences tableDeComp
    readyop = readyopbis.copy() #Liste de priorités
    cycle = 0 #temps dans l'emploi du temps
    n = len(tableDeComp) # nb de patients
    #J = [5, 3, 2, 5, 5, 3, 4, 4, 3, 4]
    m = 6 # nombre de machines

    inflightop = set() # opérations en cours d'exécution
    readymach = {k for k in range(1, m+1)} # machines disponibles

    Liste_skills=[[] for i in range(6)] #la seconde liste contiendra en premier indice l'opération effectuée et en second indice le patient pour cette opération

    M = tableDeComp
    M1 = deepcopy(M)

    inflightpatient = set() # patients actuellement en opération

    while len(readyop) + len(inflightop) > 0:

        for op in readyop.copy(): #On regarde les opérations disponibles
            i = op[0] - 1
            j = op[1] - 1
            #print(set(M[i][j]) <= readymach)
            if i not in inflightpatient and set(M[i][j]) <= readymach and (i+1, j) not in readyop: #Si le patient et les compétences sont disponibles et que le critère de précédence est respecté:
                readyop.remove(op)   #On met à jour les variables
                inflightop.add(op)
                inflightpatient.add(i)
                #print('+', op)
                for k in set(M[i][j]):
                    readymach.remove(k)

                for skill in M[i][j]:
                    Liste_skills[skill-1].append((j+1,i+1))  #On planifie l'opération

        cycle = cycle + 1

        for comp in readymach:
            Liste_skills[comp-1].append((-1,-1))
        for op in inflightop.copy(): #On dit que le patient et/ou la machine sont libérés si c'est le cas, et on met à jour les variables
            i = op[0] - 1
            j = op[1] - 1
            doublon = False
            record = 0
            for k in M1[i][j].copy():
                #print("k: ", k, "record : ", record)
                if record != k:
                    readymach.add(k)
                    M1[i][j].remove(k)
                    if not doublon:
                        inflightop.discard(op)
                        inflightpatient.discard(i)
                        #print('-', op)
                    #print(M1[i][j])
                else: # si on utilise la même compétence sur plusieurs cycles consécutifs pour une opération donnée
                    inflightop.add(op)
                    inflightpatient.add(i)
                    #print('+', op)
                    readymach.discard(k)
                    doublon = True
                record = k

    return Liste_skills

s0 = AlgoListe(readyop, tableDeComp)

newS = create_filled_empty_sol(s0,150)

##Voisinnage

def Vois(listeSol, ListeMvtTabou, best, tableDeComp): #crée le voisinage d'une solution déterminée par la liste de priorités utilisée dans l'algo de liste, en utilisant l'algo de liste
#Voisinage grossier, obtenu en permutant une seule fois deux opérations successives, ce qui peut s'avérer coûteux en pratique
    res = []
    for k in range(len(listeSol)-1):
        listeSolbis = deepcopy(listeSol)
        a = listeSolbis[k]
        b = listeSolbis[k+1]
        listeSolbis[k] = b
        listeSolbis[k+1] = a
        solVois = AlgoListe(listeSolbis, tableDeComp)
        mouvement = (a,b) # le mouvement effectué pour ce voisin correspond à un échange de a et de b dans la liste
        if (mouvement not in ListeMvtTabou) or (fbis(solVois) < best): #on vérifie qu'il ne s'agit pas d'une solution créée à partir d'un mouvement Tabou, et si elle l'est, on la prend quand même seulement si elle vérifie la fonction d'aspiration
            # if fbis(solVois) < best:
                #print("critère d'aspiration activé")
            if admissible(solVois):
                res.append((solVois,listeSolbis, mouvement))
            else:
                print("voisin non admissible, numéro:",k)
        #else:
            #print("taboued",k)
    return res #renvoie un triplet (solution, liste_associée, mouvement_emprunté)



def N(s): # Fonction donnant cette fois-ci le voisinage d'un planning et non d'une liste : planning obtenu en permutant (une seule fois) deux opérations successives dans s pour une compétence
    res = []
    m = len(s)
    for k in range(m):
        for l in range(len(s[k])-1):
            s1 = deepcopy(s)
            s1[k][l] = s[k][l+1] # on permute deux opérations successives dans le planning s l+1 et l+2 pour la compétence k+1
            s1[k][l+1] = s[k][l]
            if admissible(s1):
                res.append(s1)
    return res

##Algo tabou

def tabu(listeS0, tableDeComp, maxSize = 20):
    # maxSize: taille max de la liste taboue

    nbiter = 0
    T = [] #liste des mouvements tabous
    meil_iter = 0
    nbmax = 50 #nombre d'itérations maximal entre deux sélections de solution optimale

    listeS = listeS0.copy()
    s = AlgoListe(listeS, tableDeComp) #solution initiale
    # print("score initial: ", fbis(s))

    listeSBest = listeS0.copy() # meilleure solution temporaire
    sBest = AlgoListe(listeSBest, tableDeComp)
    best = fbis(sBest) #score correspondant

    if not admissible(s):
        print("Solution initiale non admissible")

    else :

        while nbiter - meil_iter < nbmax:
            nbiter += 1
            Voisinage = Vois(listeS, T, best, tableDeComp)
            if len(Voisinage) == 0:
                print("erreur : aucun voisin")
            else:
                (sMin,listeSMin,mouvementMin) = Voisinage[rd.randrange(len(Voisinage))] #On choisit arbitrairement le premier voisin: ceci est important notamment si de nombreux voisins ont la même valeur en fbis() qui est minimale pour le voisinage auquel ils appartiennent.
                min = fbis(sMin)
                for vois in Voisinage: #On cherche le meilleur voisin de s (les mouvements tabous sont directement exclus dans la fonction Vois(), en prenant également en compte le critère d'aspiration)
                    (solVois,listeSolVois,mouvement) = vois
                    if fbis(solVois) < min: #On choisit la meilleure solution restante
                        (sMin,listeSMin,mouvementMin) = (solVois,listeSolVois,mouvement)
                        min = fbis(solVois)
                (a,b) = mouvementMin
                T.append((b,a)) #On ajoute le mouvement inverse à la liste des mouvements tabous
                if len(T) > maxSize:
                    del T[0]
                    #print("taille limite atteinte")

                s = sMin
                listeS = listeSMin
                if fbis(s) < fbis(sBest):
                    sBest = s
                    listeSBest = listeS
                    best = fbis(sBest)
                    meil_iter = nbiter
                #print(best)
                    #print(meil_iter, star)

        #trace_Sol(sBest)
    #print("best",best)
    return (sBest, listeSBest)


listeNewS = readyop.copy()
rd.shuffle(listeNewS)
newS = AlgoListe(listeNewS, tableDeComp)
#(a,b) = tabu(listeNewS, tableDeComp) #test de la fonction tabu
#trace_Sol(a)

## appels successifs de tabou sur des tableDeComp de tailles différentes


def appels(tableDeComp, maxSize = 20, nbOpMax = 60): #on part d'une tableDeComp initiale pour former une famille de plusieurs Tables de Compétences obtenues successivement en rajoutant des opérations générées aléatoirement
    global n
    n = 10
    nbOpMin = 0
    TableCompActuelle = deepcopy(tableDeComp)
    TableCompActuelle.append([])
    for patient in range(len(tableDeComp)):
        nbOpMin += len(tableDeComp[patient])
    (limTinf,limTsup) = (1,4) #nombres min et max de tâches par opération (pour l'instant, pas plus de deux fois la même machine par opération)
    nbOpPerPatient = rd.randint(2,5)
    liste_score_tabou = []
    liste_score_ini = []

    for nbOp in range (nbOpMin,nbOpMax, 3): #On calcule le nombre d'opérations
        nbmach = rd.randint(1,4)
        L = []
        while len(L)!= nbmach:
            mach = rd.randint(1,4)
            s = 0
            for i in range(len(L)):
                if mach in L:
                    s+=1
            if s <= 1:
                L.append(mach)

        nbOpLastPatient = len(TableCompActuelle[-1])
        if nbOpLastPatient == nbOpPerPatient:
            nbOpPerPatient = rd.randint(2,5) #On met à jour pour le prochain patient
            TableCompActuelle.append([])

        TableCompActuelle[-1].append(L)
        s = 0
        for patient in range(len(TableCompActuelle)):
            s += len(TableCompActuelle[patient])

        n = len(TableCompActuelle)
        ListeIni = ListeInitiale(TableCompActuelle)
        liste_score_ini.append(fbis(AlgoListe(ListeIni, TableCompActuelle)))
        liste_score_tabou.append(fbis(tabu(ListeIni, TableCompActuelle, maxSize)[0]))
        print (nbOp+1, liste_score_ini[-1], liste_score_tabou[-1])

    return liste_score_ini,liste_score_tabou

def moyennage(n = 10, maxSize = 20): #On fait une moyenne en utilisant la fonction appels()
    listeMoyListe = np.zeros(8)
    listeMoyTabou = np.zeros(8)
    print("n", n)
    for i in range(n):
        (liste_score_ini,liste_score_tabou) = appels(tableDeComp, maxSize)
        listeMoyListe += np.array(liste_score_ini)
        listeMoyTabou += np.array(liste_score_tabou)
    listeMoyListe = listeMoyListe/n
    listeMoyTabou = listeMoyTabou/n
    return listeMoyListe, listeMoyTabou


#(liste_score_ini,liste_score_tabou) = appels(tableDeComp)
#(listeMoyListe, listeMoyTabou) = moyennage()

plt.clf()

# liste_x = [x for x in range(39,61)]
# liste_score_ini_record = [663, 975, 1234, 1619, 1731, 1940, 2289, 2289, 2917, 3972, 4446, 4938, 5611, 5319, 5564, 6458, 7052, 6881, 6553, 6819, 7426, 7241]
# liste_score_tabou_record = [379, 334, 884, 944, 1082, 1384, 1700, 1937, 1674, 1290, 2405, 3240, 2535, 2655, 3659, 3532, 4931, 4482, 5170, 4842, 5028, 6506]
# rapport = [(liste_score_ini_record[i]-liste_score_tabou_record[i])/liste_score_ini_record[i] for i in range(len(liste_score_ini_record))]

# plt.plot(liste_x,rapport)
# plt.plot(liste_x,liste_score_ini_record)
# plt.plot(liste_x,liste_score_tabou_record)


# listeMoyListe_record = [ 823.8, 1058.9, 1360.2, 1522.6, 1837.4, 2212.7, 2574.1, 2750.5]
# listeMoyTabou_record = [ 378.7,  491.9,  665.1,  936.7,  955. , 1185.6, 1449.3, 1593. ]
# liste_x = [x for x in range(39,61, 3)]
# rapport = [(listeMoyListe_record[i]-listeMoyTabou_record[i])/listeMoyListe_record[i] for i in range(len(listeMoyTabou_record))]
# plt.plot(liste_x,listeMoyListe_record)
# plt.plot(liste_x,listeMoyTabou_record )
# # plt.plot(liste_x,rapport)

#plt.show()

def moyenne_variation_longueur_tabou(it=10): #C'est la fonction qui permet d'obtenir les courbes donnant l'efficacité de l'algorithme tabou en fonction de la longueur de la liste tabou (définies dans Llong), avec it le nombre d'itérations qui permet d'obtenir un résultat correspondant à une moyenne de m = it itérations.
    global n
    def appelsbis(tableDeComp, maxSize, listeTableComp, nbOpMax = 60): #on part d'une tableDeComp initiale
        global n
        n = 10
        nbOpMin = 0
        for patient in range(len(tableDeComp)):
            nbOpMin += len(tableDeComp[patient])

        nbOpPerPatient = rd.randint(2,5)
        liste_score_tabou = []
        liste_score_ini = []
        incr = 9
        for nbOp in range (nbOpMin,nbOpMax): #On calcule le nombre d'opérations
            incr+=1

            TableCompActuelle = deepcopy(listeTableComp[nbOp-nbOpMin])
            # s = 0
            # for patient in range(len(TableCompActuelle)):
            #     s += len(TableCompActuelle[patient])
            if incr >=3:
                n = len(TableCompActuelle)
                ListeIni = ListeInitiale(TableCompActuelle)
                liste_score_ini.append(fbis(AlgoListe(ListeIni, TableCompActuelle)))
                liste_score_tabou.append(fbis(tabu(ListeIni, TableCompActuelle, maxSize)[0]))
                incr = 0
            print (nbOp+1, liste_score_ini[-1], liste_score_tabou[-1])

        return liste_score_ini,liste_score_tabou
    Llong=[0,5,20,50,100]
    liste_x = [x for x in range(39,61, 3)]
    plt.clf()

    LlisteMoyTabou = [np.zeros(8) for i in range(len(Llong))]
    for i in range(it):
        print ("étape numéro ", i)
        listeTableComp = []
        nbOpMin = 0
        TableCompActuelle = deepcopy(tableDeComp)
        TableCompActuelle.append([])
        for patient in range(len(tableDeComp)):
            nbOpMin += len(tableDeComp[patient])
        (limTinf,limTsup) = (1,4) #nombres min et max de tâches par opération (pour l'instant, pas plus de deux fois la même machine par opération)
        nbOpPerPatient = rd.randint(2,5)
        for nbOp in range (nbOpMin,60): #On calcule le nombre d'opérations
            nbmach = rd.randint(1,4)
            L = []
            while len(L)!= nbmach:
                mach = rd.randint(1,4)
                s = 0
                for j in range(len(L)):
                    if mach in L:
                        s+=1
                if s <= 1:
                    L.append(mach)

            nbOpLastPatient = len(TableCompActuelle[-1])
            if nbOpLastPatient == nbOpPerPatient:
                nbOpPerPatient = rd.randint(2,5) #On met à jour pour le prochain patient
                TableCompActuelle.append([])

            TableCompActuelle[-1].append(L)
            listeTableComp.append(deepcopy(TableCompActuelle))
        for k in range (len(Llong)):
            longueur = Llong[k]
            (liste_score_ini,liste_score_tabou) = appelsbis(tableDeComp, longueur, listeTableComp, nbOpMax = 60)
            LlisteMoyTabou[k] += np.array(liste_score_tabou)
    for k in range(len(Llong)):
        LlisteMoyTabou[k] = LlisteMoyTabou[k]/it
        plt.plot(liste_x, LlisteMoyTabou[k], label="longueur = "+str(Llong[k]))

    plt.title("Evolution de la performance de l'algorithme Tabou en fonction de la longueur de la liste tabou")
    plt.legend()
    plt.xlabel("nombre d'opérations à ordonnancer'")
    plt.ylabel("valeur de la fonction d'évaluation'")
    plt.show()





