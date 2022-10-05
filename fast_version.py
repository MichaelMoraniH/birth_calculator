import matplotlib.pyplot as plt
import torch
###### Tune your parameters here: ######
num_of_generations = 30       # Number of next generations to calculate (1 is the generation of my children)
num_of_children = 1           # Number of my children. for [0,1,2,3,4] or "random" for common man with unknown
                              # number of children, who is divided like the community (but married)
RADIUS = 22000                # The maximum effect to calculate

# All the values k you want to see the probability for the effect to be k:
specific_values = [-7, 0]
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calc_net_effect(generations, num_of_children="random", COMP_FLAG=False):
    # "random" for an average person, we don't know in advance how many children he will have
    if generations==1:
        rand_dens = (torch.tensor(NUM_OF_CHILDREN).to(device), 0, 4)
        if num_of_children in [0, 1, 2, 3, 4]:
            my_dens =  (torch.tensor([1.0]).to(device), num_of_children, num_of_children)
        else:
            my_dens = rand_dens
        return my_dens, rand_dens

    one_child_effect = calc_weighted_effect(generations-1, COMP_FLAG=COMP_FLAG)
    consider_singleness(one_child_effect)

    ebnoc = dict()  # effect by num of children

    ebnoc[0] = (torch.tensor([1.0]).to(device), 0, 0)
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
    for k in range(len(dict[0])):
        dict[0][k]*= (1-SINGLENESS)
    dict[0][-dict[1]]+= SINGLENESS


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

    tens1, low1, high1 = dict1
    tens2, low2, high2 = dict2

    if not PLUS_OR_MINUS:
        tens2 = torch.flip(tens2, [0])
        low_temp = -high2
        high2 = -low2
        low2 = low_temp

    range1 = high1-low1

    firs_line = torch.cat([tens2, torch.zeros(range1).to(device)])
    conv = [firs_line]

    for i in range(1, range1+1):
        conv.append(torch.cat([torch.tensor([0]).to(device), conv[i - 1][0:-1]]))

    conv = torch.stack(conv)
    new_tens = tens1 @ conv

    # Correction of inaccuracies resulting from computer memory limitations
    s1 = tens1.sum()
    s2 = tens2.sum()
    s = new_tens.sum()

    if s > s1*s2 or s < s1*s2:
        new_tens = new_tens *s1*s2 / s

    if COMP_FLAG:
        new_tens = new_tens / new_tens.sum()

    new_low = low1+low2
    new_high = high1+high2

    # Lower clipping
    if new_low<-RADIUS-1:
        dif = -RADIUS - new_low
        if COMP_FLAG:
            lower_sum = new_tens[:dif-1].sum()
            new_tens[dif-1] = new_tens[dif-1]+lower_sum

        new_tens=new_tens[dif-1:]
        new_low=-RADIUS-1

    # Upper clipping
    if new_high>RADIUS+1:
        dif = new_high - RADIUS
        if COMP_FLAG:
            upper_sum = new_tens[-dif+1:].sum()
            new_tens[-dif] = new_tens[-dif]+upper_sum

        new_tens=new_tens[:-dif+1]
        new_high=RADIUS+1

    return (new_tens, new_low, new_high)


def adding_probabilities(dict1, dict2, factor1, dict3=None, factor2=None, dict4=None, factor3=None, dict5=None, factor4=None):
    # Compute the probability of x=k according to its probability in case dict1 or in case dict2.
    # factor1: the probability of case dict1
    if not dict3:
        factor2 = 1-factor1

    tens1, low1, high1 = dict1
    tens2, low2, high2 = dict2

    if low1<low2:
        dif = low2-low1
        tens2 = torch.cat([torch.zeros(dif).to(device), tens2])
        low2=low1
    elif low2<low1:
        dif = low1 - low2
        tens1 = torch.cat([torch.zeros(dif).to(device), tens1])
        low1 = low2

    if high2<high1:
        dif = high1-high2
        tens2 = torch.cat([tens2,torch.zeros(dif).to(device)])
        high2=high1
    elif high1<high2:
        dif = high2-high1
        tens1 = torch.cat([tens1,torch.zeros(dif).to(device)])
        high1=high2

    tens1 = tens1 * factor1 / (factor1 + factor2)
    tens2 = tens2 * factor2 / (factor1 + factor2)

    new_tens = tens1+tens2
    new_dict = (new_tens, low1, high1)

    if dict3:
        return adding_probabilities(new_dict, dict3, factor1+factor2, dict4, factor3, dict5, factor4)

    return new_dict


def calc_range(probs, low="-inf", high="inf", COMP_FLAG=0):
    if low == "-inf":
        left = -RADIUS-1
    else:
        left = max([-RADIUS-COMP_FLAG, low])

    if high=="inf":
        right = RADIUS+1
    else:
        right = min([RADIUS+COMP_FLAG, high])

    return probs[0][left-probs[1]:right-probs[1]+1].sum().item()


def print_values_and_ranges(effects, COMP_FLAG=0):
    left = max([effects[1], -RADIUS])
    right = min([effects[2], RADIUS])
    if PRINT_AS_ARRAY:
        print([effects[0][i-effects[1]].item() for i in specific_values if i>=left and i<=right])
    else:
        for i in specific_values:
            if i>=left and i<=right:
                print(f"probability for effect={i}:", effects[0][i-effects[1]].item())

    print()
    for r in ranges:
        print(f"probability for {r[0]}<=effect<={r[1]}:", calc_range(effects, r[0], r[1], COMP_FLAG))


effects_naiv = calc_weighted_effect(num_of_generations, num_of_children)
effects_comp = calc_weighted_effect(num_of_generations, num_of_children, True)

print("Precision:", effects_naiv[0].sum().item(), "\n")
print("------- lower bound -------")
print_values_and_ranges(effects_naiv)
print()
print("------- upper bound -------")
print_values_and_ranges(effects_comp, 1)

left = max(-RADIUS, effects_naiv[1])
right = min(RADIUS, effects_naiv[2])

dict_naiv = effects_naiv[0]
dict_comp = effects_comp[0]

if left!=effects_naiv[1]:
    dict_naiv = dict_naiv[1:]
    dict_comp = dict_comp[1:]

if right!=effects_naiv[2]:
    dict_naiv = dict_naiv[:-1]
    dict_comp = dict_comp[:-1]

plt.plot(range(left, right+1), dict_naiv.tolist(), label="lower bound")
plt.plot(range(left, right+1), dict_comp.tolist(), label = "upper bound")
plt.legend(["lower bound", "upper bound"])
plt.show()