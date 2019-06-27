import pickle
import matplotlib.pyplot as plt
import numpy as np

from SnakeGame import SnakeGame
from NeuralNet import NeuralNet
import GeneticAlg

#---------------- Analysis

def showEvolution(NNsave):
	rewardL = []
	for k in sorted(list(NNsave.keys())):
		G = SnakeGame(trainingGame = False)
		G.NNetPlayShow(NNsave[k][0][0], generation = k)
		rewardL.append((G.reward, G.score))

	c = 0
	for k in sorted(list(NNsave.keys())):
		print('Generation {} - Game reward {}, Score {} - Generation reward {}'.format(k,rewardL[c][0],rewardL[c][1], NNsave[k][0][1]))
		c += 1


def meanAndStdReward(NNsave, genNb):
	#Calculate the average and standard deviation of the reward of a generation in NNsave
	L = NNsave[genNb]
	rewards = []
	for NN, R in L:
		rewards.append(R)
	return np.mean(rewards), np.std(rewards)

def plotRewardsFromTraining(NNsave):
	#Plot the rewards calculated at the generation process
	x = sorted(list(NNsave.keys()))
	y = [] #Average
	yerr = [] #Standard deviation
	maxy = [] #Max reward
	miny = [] #Min reward

	for gen in x:
		mean, std = meanAndStdReward(NNsave, gen)
		y.append(mean)
		yerr.append(std)
		maxy.append(NNsave[gen][0][1])
		miny.append(NNsave[gen][-1][1])


	fig, ax = plt.subplots()

	meanPlot = ax.errorbar(x, y,
	            yerr=yerr,
	            fmt='-o')
	maxPlot = ax.plot(x, maxy, 'ro--')
	minPlot = ax.plot(x, miny, 'go--')

	plt.legend((meanPlot[0], maxPlot[0], minPlot[0]), ('Average reward', 'Maximum reward', 'Minimum reward'))

	ax.set_xlabel('Generation #')
	ax.set_ylabel('Reward (higher the better)')
	ax.set_title('Evolution of the generations at training')

	#plt.show()
	fig.savefig('trainingResults.png', dpi = 300)


def testSnake(model, meanOn):
	rewards = []
	maxReward = None
	minReward = None
	games = [SnakeGame(trainingGame = False) for _ in range(meanOn)]

	for G in games:
		G.NNetPlay(model)
		rewards.append(G.reward)
		if maxReward is None:
			maxReward = G.reward
		else:
			maxReward = max(maxReward, G.reward)

		if minReward is None:
			minReward = G.reward
		else:
			minReward = min(minReward, G.reward)

	return np.mean(rewards), np.std(rewards), maxReward, minReward

def plotRewardsFromTesting(NNsave, meanOn = 40):
	#Plot the rewards of the snakes on previously unseen games.
	#We only select the best snake for each generation as the representative.
	#meanOn is the number of games we test them on.

	x = sorted(list(NNsave.keys()))
	y = [] #Average
	yerr = [] #Standard deviation
	maxy = [] #Max reward
	miny = [] #Min reward

	for gen in x:
		mean, std, maxReward, minReward = testSnake(NNsave[gen][0][0], meanOn)
		y.append(mean)
		yerr.append(std)
		maxy.append(maxReward)
		miny.append(minReward)

	fig, ax = plt.subplots()

	meanPlot = ax.errorbar(x, y,
	            yerr=yerr,
	            fmt='-o')
	maxPlot = ax.plot(x, maxy, 'ro--')
	minPlot = ax.plot(x, miny, 'go--')

	plt.legend((meanPlot[0], maxPlot[0], minPlot[0]), ('Average reward', 'Maximum reward', 'Minimum reward'))

	ax.set_xlabel('Generation #')
	ax.set_ylabel('Reward (higher the better)')
	ax.set_title('Testing the best snake of each generation over {} unseen games'.format(meanOn))

	#plt.show()
	fig.savefig('testingResults.png', dpi = 300)

if __name__ == "__main__": #Testing
	#----- Recover last training
	try:
		NNsave = pickle.load(open( "NNsave.pickle", "rb" ))
	except:
		NNsave = {}
	
	#----- Train if necessary
	generationNb = 60
	populationPerGen = 1000
	
	GeneticAlg.geneticAlgorithm(NNsave, populationPerGen, generationNb, meanOn = 10) #NNsave2.pickle
	
	#----- Save the work
	pickle.dump(NNsave, open("NNsave.pickle", "wb"))

	#----- Select only the generation we want to see for the showEvolution()
	import random as rd

	partialNNsave = {}
	showNumber = 10
	partialNNsave[1] = NNsave[1]
	partialNNsave[generationNb] = NNsave[generationNb]
	for _ in range(showNumber-2):
		genSelected = rd.randint(2, generationNb-1)
		partialNNsave[genSelected] = NNsave[genSelected]

	#showEvolution(partialNNsave)

	G = SnakeGame(trainingGame = False)
	model = NNsave[59][0][0]
	G.NNetPlayShow(model)

	#----- Plot the rewards
	#plotRewardsFromTraining(NNsave)
	#plotRewardsFromTesting(NNsave)

