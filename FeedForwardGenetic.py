
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.ops import rnn, rnn_cell
import pickle
from random import randint
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



class Member(object):
    def __init__(self, nHid, nHid2, nHid3, epoch, learningrate, batchsize, acc):
        self.nHid = nHid
        self.nHid2 = nHid2
        self.nHid3 = nHid3
        self.epoch = epoch
        self.learningrate = learningrate
        self.batchsize = batchsize
        self.acc = acc
        self.trainacc = 0
    def __lt__(self, acc):
        return self.acc

params = {
    'nHid' : 60,
    'dropout' : 10,
    'epoch' : 500,
    'learningrate' : 100,
    'batchsize' : 4,
}

def breed(parents, numkids):
    kids = []
    for i in range(0, numkids):
        nHid = int((parents[0].nHid + parents[1].nHid) /2) + randint(-2,2)
        nHid2 = int((parents[0].nHid2 + parents[1].nHid2) /2) + randint(-2,2)
        nHid3 = int((parents[0].nHid3 + parents[1].nHid3) /2) + randint(-2,2)
        epoch = int((parents[0].epoch + parents[1].epoch) /2) + randint(-2,2)
        learningrate = ((parents[0].learningrate + parents[1].learningrate) /2)
        batchsize = (( (parents[0].batchsize) / 32 + (parents[1].batchsize / 32)  /2) * 32)

        kids.append(Member(nHid, nHid2, nHid3, epoch, learningrate, batchsize, 0 ))
    kids.extend(initalizepop(5))
    return kids
        


def evolve(population, generation, ev):
    numkids = 3
    numsurvive = 10
    population.sort(key = lambda i: i.acc, reverse=True)
    top = population[:6]
    print('[+]------------------  GENERATION ', generation,' RESULTS ------------------[+]')
    print('Top 5 member accuracy: ', [top[x].acc for x in range(0, 4)])
    print('Remaining Generations: ', ev + 1 - generation)
    
    newpop = []
    for i in range(0, numsurvive):
        for x in range(i + 1, numsurvive + 1):
            parents = [ population[i],population[x] ]
            kids = breed(parents, numkids) ############# review when sober
            newpop.extend(kids)
    print(len(newpop), ' Members added to new population')
    print('[+]------------------  GENERATION ', generation,' RESULTS ------------------[+]')
    return newpop


def runpop(population, training, val):
    for model in range(0, len(population)):
        print('[*] Model: ', model, 'Evaluating')
        print('Model Attributes:')
        print('nHid: ', population[model].nHid)
        print('nHid2: ', population[model].nHid3)
        print('nHid2: ', population[model].nHid3)
        print('epochs: ', population[model].epoch)
        print('learningrate: ', population[model].learningrate)
        print('batchsize: ', population[model].batchsize)
        #population[model].acc  = test(population, model, training, val)
        try: 
            population[model].acc  = test(population, model, training, val)
            print('[+] Model: ', model, 'Completed\n')
        except:
            print('[!!!!] MODEL FAILED\n')
    return population

def initalizepop(popsize):
    pop = []
    while len(pop) <= popsize:
        nHid = randint(1, params['nHid'])
        nHid2 = randint(1, params['nHid'])
        nHid3 = randint(1, params['nHid'])
        epoch = randint(3, params['epoch'])
        learningrate = .001  #currently fixed but could be made dynamic just like above
        batchsize = randint(1, params['batchsize']) * 32
        pop.append(Member(nHid, nHid2, nHid3, epoch, learningrate, batchsize, 0 ))
    return pop


def test(population, modelindex, training, val):
    model = population[modelindex]
    print('Training...')
    accuracy = trainmodel(model, training, val)
    print('[!] Accuracy AVG: ', accuracy)
    return accuracy

def trainmodel(model, training, val):
    X_train = training['X']
    Y_train = training['Y']
    X_val = val['X']
    Y_val = val['Y']
    acclist = []
    inputs = 15
    layer_1_neurons = model.nHid
    layer_2_neurons = model.nHid2
    layer_3_neurons = model.nHid3
    output_neurons = 1

    learning_rate = model.learningrate
    epochs = model.epoch
    batch_size = model.batchsize

    #Creating place holders for inputs and outputs
    X = tf.placeholder(dtype=tf.float32, shape=[None, inputs])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, output_neurons])

    #defining layer attributes
    h_1_layer_weights = tf.Variable(tf.random_normal([inputs, layer_1_neurons]))
    h_1_layer_bias = tf.Variable(tf.random_normal([layer_1_neurons]))

    h_2_layer_weights = tf.Variable(tf.random_normal([layer_1_neurons, layer_2_neurons]))
    h_2_layer_bias = tf.Variable(tf.random_normal([layer_2_neurons]))

    h_3_layer_weights = tf.Variable(tf.random_normal([layer_2_neurons, layer_3_neurons]))
    h_3_layer_bias = tf.Variable(tf.random_normal([layer_3_neurons]))

    output_weights = tf.Variable(tf.random_normal([layer_3_neurons, output_neurons]))
    output_bias = tf.Variable(tf.random_normal([output_neurons]))

    #defining layer calculators and flow
    #REMEMBER Weights
    l1_calc = tf.nn.relu(tf.add(tf.matmul(X, h_1_layer_weights), h_1_layer_bias))
    l2_calc = tf.nn.relu(tf.add(tf.matmul(l1_calc, h_2_layer_weights), h_2_layer_bias))	
    l3_calc = tf.nn.relu(tf.add(tf.matmul(l2_calc, h_3_layer_weights), h_3_layer_bias))
    output_calc = tf.add(tf.matmul(l3_calc, output_weights), output_bias)

    #defining training, cost, and accuracy calculations
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_calc, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    predicted_class = tf.equal(tf.round(tf.nn.sigmoid(output_calc)), tf.round(Y))
    accuracy = tf.reduce_mean(tf.cast(predicted_class, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  #initalizes the varibles in prep for training


        for e in range(0, epochs):
            epoch_loss = 0 
            shuffle_set = np.random.permutation(np.arange(len(Y_train))) #randomly shuffles data for each epoch to combat overfitting
            X_train = X_train[shuffle_set]
            Y_train = Y_train[shuffle_set]



            for i in range(0, len(Y_train) // batch_size):
                #print("Batch: {0}".format(str(i) + "/" + str(len(Y_train) // batch_size)), end="\r")
                start = i * batch_size
                batch_x = X_train[start:start + batch_size]
                batch_y = Y_train[start:start + batch_size]
                _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y}) #runs the batch in the optimizer and gets the cost
                epoch_loss += c #compounds cost for epoch
        acclist.append(accuracy.eval({X: X_val, Y: Y_val}))
        #print(acclist[-1])
    acclist.reverse()
    return (acclist[-1])
def controlla():
    training = pickle.load(open('training.dat', 'rb'))
    testing = pickle.load(open('testing.dat', 'rb'))
    val = pickle.load(open('validation.dat', 'rb'))
    print("Genetic Running... \n\n")
    totalevolutions = 50
    populationsize = 100
    population = initalizepop(populationsize)
    pops = {}
    print('[+] Population Initalized')
    for ev in range(0, totalevolutions):
        population = runpop(population, training, val)
        pops[ev] = population
        population = evolve(population, ev, totalevolutions)
        pickle.dump(pops, open('GenResults.dat', 'wb'), -1)



if __name__ == "__main__":
    controlla()