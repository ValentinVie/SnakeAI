import pickle


from SnakeGame import SnakeGame
from NeuralNet import NeuralNet
from GeneticAlg import geneticAlgorithm

#---------------- Analysis

def showEvolution(NNsave):
	rewardL = []
	for k in list(NNsave.keys()):
		G = SnakeGame(trainingGame = False)
		G.NNetPlayShow(NNsave[k][0][0], generation = k)
		rewardL.append((G.reward, G.score))
	for k in list(NNsave.keys()):
		print('Generation {} - Game reward {}, Score {} - Generation reward {}'.format(k,rewardL[k-1][0],rewardL[k-1][1], NNsave[k][0][1]))


def plotRewardsFromTraining(NNsave):
	#Plot the rewards calculated at the generation process
	pass

def plotRewardsFromTesting(NNsave):
	#Plot the rewards of the snakes on previously unseen games
	pass



if __name__ == "__main__": #Testing
	try:
		NNsave = pickle.load(open( "NNsave.pickle", "rb" ))
	except:
		NNsave = {}

	geneticAlgorithm(NNsave, 500, 15, meanOn = 10)
	showEvolution(NNsave)


	pickle.dump(NNsave, open("NNsave.pickle", "wb"))