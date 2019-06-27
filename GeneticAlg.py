import random as rd
from tqdm import tqdm
from multiprocessing import Process, Queue

from SnakeGame import SnakeGame
from NeuralNet import NeuralNet

#---------------- Genetic Algorithm


def evaluate(newGen, meanOn, Q):
	rewards = []
	N = len(newGen)
	for _ in range(meanOn):
		games = [SnakeGame() for _ in range(N)]
		for k in range(N):
			games[k].NNetPlay(newGen[k])
			if len(rewards) != N:
				rewards.append((newGen[k], games[k].reward*1/meanOn))
			else:
				rewards[k] = (rewards[k][0], rewards[k][1]+rewards[k][1]*1/meanOn)

	Q.put(rewards) #Sorted by reward later in the main function

def evaluateMultithreaded(newGen, meanOn, numberOfProcess = 6):
	rewards = []
	processList = []
	Q = Queue()
	N = len(newGen)

	for i in range(numberOfProcess-1):
		#Creating the process & starting  
		p = Process(
		        target=evaluate,
		        args=(newGen[i*int(N/numberOfProcess):(i+1)*int(N/numberOfProcess)], meanOn, Q))
		processList.append(p)
		p.start()
	
	#The last process takes the remainder
	p = Process(
		        target=evaluate,
		        args=(newGen[(numberOfProcess-1)*int(N/numberOfProcess):], meanOn, Q))
	processList.append(p)
	p.start()

	#Extract the rewards
	for i in range(len(processList)):
		rewards.extend(Q.get())
	            
	# Wait for the processes to finish...
	for p in processList:
		p.join()

	return sorted(rewards, key=lambda x: x[1], reverse=True) #Sort by reward



def geneticAlgorithm(NNsave, N, generation, meanOn = 5, keepBest = 0.1, mutate = 0.3, merge = 0.3):
	#NNsave: Dictionnary with all the different generations.
	#N: Number of NN kept in a list each iteration.
	#generation: The number of generation our system will have.
	#meanOn: Run the NN on meanOn # of games.
	#keepBest: % of NN we don't modify and keep for future gen.
	#mutate: % of NN where we mutate 1-2 parameters.
	#merge: % of NN we merge together.
	#rand: We pick all other NN at random
	
	keys = list(NNsave.keys())
	
	#Initialization for the first generation
	if len(keys) == 0:
		print("Generating the first generation...")
		NNsave[1] = evaluateMultithreaded([NeuralNet() for _ in range(N)], meanOn) #Sorted by reward
		keys = [1]

	#Check if we need to compute something or if all gen are already calculated
	maxGen = max(keys)
	if maxGen >= generation:
		return

	#Main Loop for the creation of the generations...
	lastGen = NNsave[maxGen]
	for genNb in tqdm(range(maxGen, generation)): #tqdm
		newGen = NNsave[genNb][:int(N*keepBest)]
		newGen = [x[0].copy() for x in newGen]#take only the NN

		mergeL = NNsave[genNb][int(N*mutate):int(N*(mutate+merge))]
		mergeL = [x[0].copy() for x in mergeL]#take only the NN

		# model = newGen[0]
		# G = SnakeGame()
		# G.NNetPlay(model)
		# print('------BEGIN-------', G.reward, model.weights1[0], model.weights1.shape)

		#----------- mutation
		for _ in range(int(N*mutate)):
			NNcpy = newGen[rd.randint(0,int(N*keepBest)-1)].copy()
			nbMutation = rd.randint(1,4)
			while nbMutation != 0:
				nbMutation -= 1
				chooseWeight = rd.randint(0,2)
				
				if chooseWeight == 0:#weights 1
					k = rd.randint(0,len(NNcpy.weights1)-1)
					NNcpy.weights1[k] += (rd.random()-0.5)*0.1
				elif chooseWeight == 1:#weights2
					k = rd.randint(0,len(NNcpy.weights2)-1)
					NNcpy.weights2[k] += (rd.random()-0.5)*0.1
				else:#weights3
					k = rd.randint(0,len(NNcpy.weights3)-1)
					NNcpy.weights3[k] += (rd.random()-0.5)*0.1
			newGen.append(NNcpy)
		
		# model = newGen[0]
		# G = SnakeGame()
		# G.NNetPlay(model)
		# print('------AFTER MUT-------', G.reward, model.weights1[0], model.weights1.shape)

		#----------- reproduction
		for NN in mergeL:
			lastNN = newGen[rd.randint(0,int(N*keepBest)-1)].copy()
			NNcpy = newGen[rd.randint(0,int(N*keepBest)-1)].copy()
			for k in range(len(NNcpy.weights1)):
				if rd.randint(0,1):
					NNcpy.weights1[k] = lastNN.weights1[k]
			for k in range(len(NNcpy.weights2)):
				if rd.randint(0,1):
					NNcpy.weights2[k] = lastNN.weights2[k]
			for k in range(len(NNcpy.weights3)):
				if rd.randint(0,1):
					NNcpy.weights3[k] = lastNN.weights3[k]
			lastNN = NNcpy
			newGen.append(NNcpy)


		#------------ random
		n = len(newGen)
		for k in range(n,N):
			newGen.append(NeuralNet())

		
		# model = newGen[0]
		# G = SnakeGame()
		# G.NNetPlay(model)
		# print('\n------BEFORE IN-------', G.reward, model.weights1[0])

		NNsave[genNb+1] = evaluateMultithreaded(newGen, meanOn) #Sorted by reward

		# model = NNsave[genNb+1][0][0]
		# G = SnakeGame()
		# G.NNetPlay(model)
		# print('\n---END----------', G.reward, model.weights1[0])

	return


if __name__ == "__main__": #Testing
	NNsave = {}
	generationNb = 5
	populationPerGen = 10
	geneticAlgorithm(NNsave, populationPerGen, generationNb, meanOn = 5)
	assert(len(NNsave) != 0)
	
	for key in range(generationNb):
		print('Generation {}'.format(key+1))
		print(NNsave[key+1])
		print('\n')
