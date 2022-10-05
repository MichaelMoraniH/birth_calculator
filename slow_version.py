import matplotlib.pyplot as plt
###### Tune your parameters here: ######
num_of_generations = 30      # Number of next generations to calculate (1 is the generation of my children)
num_of_children = "random"    # Number of my children. for [0,1,2,3,4] or "random" for common man with unknown
                              # number of children, who is divided like the community (but married)
RADIUS = 4000                # The maximum effect to calculate

# All the values k you want to see the probability for the effect to be k:
specific_values = [0]
PRINT_AS_ARRAY = True   # Print the specific values in form [a, b, ...]

# All the ranges [k, m] you want to see the probability for the effect to be from k to m.
# "-inf" for no lower limit and "inf" for no upper limit"
ranges = [
    [-10,10],
    ["-inf", -1],
    [1, "inf"],
    [-1000, 1000],
    [-5000, 5000],
    ["-inf", "inf"],
    ]

# World parameters:
NUM_OF_CHILDREN = [0.29, 0.20, 0.15, 0.18, 0.18]  # The probabilities for couple having 0, 1, 2, 3, 4 children
SINGLENESS = 0.17                                 # The probability for a person to be single

#######################
"""
The code:
"""
#######################

ADDED_COUPLE = 1 / (2-SINGLENESS)   # The probability for married person to add another couple to the world


def calc_net_effect(generations, num_of_children="random", COMP_FLAG=False):
    # "random" for an average person, we don't know in advance how many children he will have
    if generations==1:
        rand_dens = {i: NUM_OF_CHILDREN[i] for i in range(5)}
        if num_of_children in [0, 1, 2, 3, 4]:
            my_dens =  {num_of_children: 1}
        else:
            my_dens = rand_dens
        return my_dens, rand_dens

    one_child_effect = calc_weighted_effect(generations-1, COMP_FLAG=COMP_FLAG)
    consider_singleness(one_child_effect)

    ebnoc = dict()  # effect by num of children

    ebnoc[0] = {0: 1}
    ebnoc[1] = one_child_effect
    ebnoc[2] = convolution_prob(one_child_effect, one_child_effect, COMP_FLAG=COMP_FLAG)
    ebnoc[3] = convolution_prob(ebnoc[2], one_child_effect, COMP_FLAG=COMP_FLAG)
    ebnoc[4] = convolution_prob(ebnoc[3], one_child_effect, COMP_FLAG=COMP_FLAG)

    rand_dens = adding_probabilities(ebnoc[0], ebnoc[1], NUM_OF_CHILDREN[0], ebnoc[2], NUM_OF_CHILDREN[1], ebnoc[3], NUM_OF_CHILDREN[2], ebnoc[4], NUM_OF_CHILDREN[3])

    if num_of_children in [0, 1, 2, 3, 4]:
        my_dens = ebnoc[num_of_children]
    else:
        my_dens = rand_dens

    return my_dens, rand_dens


def consider_singleness(dict):
    # Given the density of effects of married person, it return the density of effects of general person, including the
    # possibility of being single.
    for k in dict.keys():
        dict[k]*= (1-SINGLENESS)
    dict[0]+= SINGLENESS


def calc_weighted_effect(generations, num_of_children="random", COMP_FLAG=False):
    my_net_dens, rand_dens = calc_net_effect(generations, num_of_children, COMP_FLAG=COMP_FLAG)
    my_gross_dens = convolution_prob(my_net_dens, rand_dens, 0, COMP_FLAG=COMP_FLAG)
    # my_net_dens: the probabilities of my effects in case I didn't add another couple to the world
    # my_gross_dens: the probabilities of my effects in case I did add another couple to the world

    my_weighted_dens = adding_probabilities(my_net_dens, my_gross_dens, ADDED_COUPLE)
    return my_weighted_dens


def convolution_prob(dict1, dict2, PLUS_OR_MINUS=1, COMP_FLAG=False):
    # Compute the probability of x1+x2(+x3+x4)=k, for each k (or x1-x2=k).
    # 1: plus, 0: minus
    res=dict()
    for k1 in dict1.keys():
        v1 = dict1[k1]
        for k2 in dict2.keys():
            v2 = dict2[k2]
            if not PLUS_OR_MINUS:
                k2 = minus(k2)
            k = clip(k1, k2, COMP_FLAG)
            if k!="undefined":
                if k not in res.keys():
                    res[k]=0
                res[k]+= v1*v2
    return res


def minus(k):
    if k == f"more than {RADIUS}": return f"less than -{RADIUS}"
    if k == f"less than -{RADIUS}": return f"more than {RADIUS}"
    return -k


def clip(k1, k2,COMP_FLAG=False):
    if k1 == f"more than {RADIUS}":
        if k2 == f"less than -{RADIUS}":
            return "undefined"
        else:
            return f"more than {RADIUS}"
    if k1 == f"less than -{RADIUS}":
        if k2 == f"more than {RADIUS}":
            return "undefined"
        else:
            return f"less than -{RADIUS}"

    if k2 == f"less than -{RADIUS}":
        return f"less than -{RADIUS}"
    if k2 == f"more than {RADIUS}":
        return f"more than {RADIUS}"

    k = k1+k2

    sign = -1
    if k>0:
        sign = 1
    if abs(k)>RADIUS:
        if COMP_FLAG: return (RADIUS+1) * sign
        else: return {-1: f"less than -{RADIUS}", 1: f"more than {RADIUS}"}[sign]

    return k


def adding_probabilities(dict1, dict2, factor1, dict3=None, factor2=None, dict4=None, factor3=None, dict5=None, factor4=None):
    # Compute the probability of x=k according to its probability in case dict1 or in case dict2.
    # factor1: the probability of case dict1
    if not dict3:
        factor2 = 1-factor1
    new_keys = list(set.union(set(dict1.keys()), set(dict2.keys())))
    new_dict = dict()
    for k in new_keys:
        prob = 0
        if k in dict1.keys():
            prob+= dict1[k] * factor1 / (factor1 + factor2)
        if k in dict2.keys():
            prob+= dict2[k] * factor2 / (factor1 + factor2)
        new_dict[k]=prob

    if dict3:
        return adding_probabilities(new_dict, dict3, factor1+factor2, dict4, factor3, dict5, factor4)

    return new_dict


def calc_range(probs, low="-inf", high="inf"):
    if low=="-inf": left = -RADIUS
    else: left = low

    if high=="inf": right = RADIUS
    else: right = high

    keys = list(range(left, right+1))

    if low == "-inf":
        keys += [f"less than -{RADIUS}"]
        keys += [-RADIUS - 1]
    if high == "inf":
        keys += [f"more than {RADIUS}"]
        keys += [RADIUS + 1]


    s = 0
    for k in keys:
        if k in probs.keys():
            s+=probs[k]
    return s


def print_values_and_ranges(effects):
    if PRINT_AS_ARRAY:
        print([effects[i] for i in specific_values])
    else:
        for i in specific_values:
            print(f"probability for effect={i}:", effects[i])

    print()
    for r in ranges:
        print(f"probability for {r[0]}<=effect<={r[1]}:", calc_range(effects, r[0], r[1]))


def collect_all_values(effects):
    keys = range(-RADIUS, RADIUS+1)
    old_keys = list(effects.keys())
    old_keys = [i for i in old_keys if type(i) == int]
    old_keys.sort()
    new_keys = []
    new_probs = []
    for key in old_keys:
        if key in keys:
            new_keys.append(key)
            new_probs.append(effects[key])
    return new_keys, new_probs

effects_naiv = calc_weighted_effect(num_of_generations, num_of_children)
effects_comp = calc_weighted_effect(num_of_generations, num_of_children, True)

print("Precision:", sum(effects_naiv.values()), "\n")
print("------- lower bound -------")
print_values_and_ranges(effects_naiv)
print()
print("------- upper bound -------")
print_values_and_ranges(effects_comp)

lower_keys, lower_probs = collect_all_values(effects_naiv)
upper_keys, upper_probs = collect_all_values(effects_comp)
plt.plot(lower_keys, lower_probs, label="lower bound")
plt.plot(upper_keys, upper_probs, label = "upper bound")
plt.legend(["lower bound", "upper bound"])
plt.show()