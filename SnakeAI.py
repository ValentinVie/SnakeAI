import random as rd
import collections as c
import curses
import time
import numpy as np
import pickle
from tqdm import tqdm


def sigmoid(x):
	return 1/(1 + np.exp(-x))

def softmax(A):  
    expA = np.exp(A)
    return expA / expA.sum()

def distance(x, y):
	return np.linalg.norm(np.array(x)-np.array(y))

def mean(L):
	return sum(L)/len(L)

#With size 10,10 the position allowed are (0,0) and (9,9)
#self.snake : Queue([tail..... head])
class SnakeGame:
	SPEED = 600
	SPEEDNN = 100

	def __init__(self, size = (10,20), maxMoves=100):
		self.size = size
		self.maxMoves = maxMoves
		self.remainingMoves = self.maxMoves
		self.snake = c.deque()
		self.snake.append((self.size[0]//2, self.size[1]//2))
		self.direction = 'DOWN'
		self.score = 0
		self.reward = 0
		self.chooseFoodPosition()

	def chooseFoodPosition(self):
		if self.score == self.size[0]*self.size[1]:
			raise NameError('Food position could not be determined.')
			return False
		self.food = ((rd.randint(1,self.size[0])-1, rd.randint(1,self.size[1])-1))
		while self.food in self.snake:
			self.food = ((rd.randint(1,self.size[0])-1, rd.randint(1,self.size[1])-1))
		return True

	def nextHead(self, nextMove):
		movementsTurns = {'UP':(0,1), 'RIGHT':(1,0), 'DOWN':(0,-1), 'LEFT':(-1,0)} 
		#dx,dy for right turns depending on the direction
		#if we go up and we turn RIGHT we do position+=(0,1)
		#if we go up and we turn LEFT we do position-=(0,1)
		movementsStraight = {'UP':(-1,0), 'RIGHT':(0,1), 'DOWN':(1,0), 'LEFT':(0,-1)}
		#dx,dy for going straight depending on the direction
		#if we go up and we go straight we do position+=(-1,0)

		directionL = ['UP', 'RIGHT', 'DOWN', 'LEFT']
		#if we go RIGHT change direction to directionL[+=1]
		#if we go LEFT change direction to directionL[-=1]

		head = self.snake[-1]
		if nextMove == 'RIGHT':
			dx,dy = movementsTurns[self.direction]
			head = (head[0]+dx, head[1]+dy)
			self.direction = directionL[(directionL.index(self.direction)+1)%4]
		elif nextMove == 'LEFT':
			dx,dy = movementsTurns[self.direction]
			head = (head[0]-dx, head[1]-dy)
			self.direction = directionL[(directionL.index(self.direction)-1)%4]
		elif nextMove == 'STRAIGHT':
			dx,dy = movementsStraight[self.direction]
			head = (head[0]+dx, head[1]+dy)

		return head


	def nextMoveUpdate(self, nextMove):
		head = self.nextHead(nextMove)
		headS = self.nextHead('STRAIGHT')
		headR = self.nextHead('RIGHT')
		headL = self.nextHead('LEFT')

		#if we choose to hit the wall and there was another way we kill the reward
		# headSGameOver = (headS in self.snake or self.size[0] == headS[0] or headS[0] < 0 or self.size[1] == headS[1] or headS[1] < 0)
		# headRGameOver = (headR in self.snake or self.size[0] == headR[0] or headR[0] < 0 or self.size[1] == headR[1] or headR[1] < 0)
		# headLGameOver = (headL in self.snake or self.size[0] == headL[0] or headL[0] < 0 or self.size[1] == headL[1] or headL[1] < 0)

		if head in self.snake or self.size[0] == head[0] or head[0] < 0 or self.size[1] == head[1] or head[1] < 0:
			# if not headSGameOver or not headRGameOver or not headLGameOver:
			# 	self.reward -= 200
			return False, self.score

		elif head == self.food: # we move to the food
			self.score += 1
			self.reward += 200
			self.snake.append(head)
			self.chooseFoodPosition()
			self.remainingMoves = self.maxMoves
		else: # we move but no food eaten
			if distance(self.snake[-1], self.food) <= distance(head, self.food): 
			# we are further from the food
				self.reward -= 1.5
			else:
			# we got closer to the food
				self.reward += 1

			self.snake.append(head)
			self.snake.popleft()
			self.remainingMoves -= 1

		if self.remainingMoves == 0:
			return False, self.score

		return True, self.score

	def hummanPlay(self):
		curses.initscr()
		screen = curses.newwin(self.size[0]+2, self.size[1]+2, 0, 0)
		screen.keypad(True)
		curses.noecho()
		curses.curs_set(False)
		screen.border(0)
		screen.nodelay(True)


		screen.addstr(0, 2, 'Score : ' + str(self.score) + ' ')
		screen.timeout(SnakeGame.SPEED)

		prevSnake = self.snake.copy()
		prevFood = self.food
		gameOn = True
		try:
			while True:
				char = screen.getch()
				if char == ord('q'):
					break
				elif char == curses.KEY_RIGHT:
					# print doesn't work with curses, use addstr instead
					gameOn, _ = self.nextMoveUpdate('RIGHT')
				elif char == curses.KEY_LEFT:
					gameOn, _ = self.nextMoveUpdate('LEFT')
				else:
					gameOn, _ = self.nextMoveUpdate('STRAIGHT')

				if not gameOn:
					break
				
				#remove
				if prevSnake[0] != self.snake[0]:
					screen.addch(prevSnake[0][0]+1, prevSnake[0][1]+1, ' ')
				prevSnake = self.snake.copy()

				if prevFood != self.food:
					screen.addch(prevFood[0]+1, prevFood[1]+1, ' ')
					prevFood = self.food
				
				#add
				headSign = None
				if self.direction == 'UP':
					headSign = '^'
				elif self.direction == 'DOWN':
					headSign = 'v'
				elif self.direction == 'RIGHT':
					headSign = '>'
				elif self.direction == 'LEFT':
					headSign = '<'

				screen.addch(self.food[0]+1, self.food[1]+1, '*')
				for x,y in self.snake:
					screen.addch(x+1, y+1, '#')
				
				x,y = self.snake[-1]
				screen.addch(x+1, y+1, headSign)

				screen.addstr(0, 2, 'Score : ' + str(self.score) + ' ')
		finally:
			# shut down cleanly
			curses.nocbreak()
			screen.keypad(0)
			curses.echo()
			curses.endwin()

		print('Final score: {}'.format(self.score))

	def NNetPlay(self, model):
		gameOn = True
		while gameOn:
			gameOn, _ = self.nextMoveUpdate(model.feedforward(self.createInputNN()))

	def NNetPlayShow(self, model, generation = None):
		curses.initscr()
		screen = curses.newwin(self.size[0]+3, self.size[1]+2, 0, 0)
		screen.keypad(True)
		curses.noecho()
		curses.curs_set(False)
		screen.border(0)
		screen.nodelay(True)

		screen.addstr(0, 2, 'Score : ' + str(self.score) + ' ')
		screen.timeout(1)

		prevSnake = self.snake.copy()
		prevFood = self.food
		gameOn = True
		speedUp = False
		try:
			while True:
				if not speedUp:
					char = screen.getch()
					if char == ord('q'):
						speedUp = True
					time.sleep(SnakeGame.SPEEDNN/1000)

				gameOn, _ = self.nextMoveUpdate(model.feedforward(self.createInputNN()))

				if not gameOn:
					break
				
				#remove
				if prevSnake[0] != self.snake[0]:
					screen.addch(prevSnake[0][0]+1, prevSnake[0][1]+1, ' ')
				prevSnake = self.snake.copy()

				if prevFood != self.food:
					screen.addch(prevFood[0]+1, prevFood[1]+1, ' ')
					prevFood = self.food
				
				#add
				headSign = None
				if self.direction == 'UP':
					headSign = '^'
				elif self.direction == 'DOWN':
					headSign = 'v'
				elif self.direction == 'RIGHT':
					headSign = '>'
				elif self.direction == 'LEFT':
					headSign = '<'

				screen.addch(self.food[0]+1, self.food[1]+1, '*')
				for x,y in self.snake:
					screen.addch(x+1, y+1, '#')
				
				x,y = self.snake[-1]
				screen.addch(x+1, y+1, headSign)

				screen.addstr(0, 2, 'Score : ' + str(self.score) + ' ')

				screen.addstr(self.size[0]+1, 2, 'Reward : ' + str(self.reward) + ' ')
				if generation:
					screen.addstr(self.size[0]+2, 2, 'Generation : ' + str(generation) + ' ')
				screen.refresh()
		finally:
			# shut down cleanly
			curses.nocbreak()
			screen.keypad(0)
			curses.echo()
			curses.endwin()

		print('Final score: {}'.format(self.score))

	def createInputNN(self):
		#Input:
		#1. Is it clear STRAIGHT ?
		#2. Is it clear LEFT ?
		#3. Is it clear RIGHT ?
		#4. Is there food STRAIGHT ?
		#5. Is there food LEFT ?
		#6. Is there food RIGHT ?
		#7. How far are we from the food in X?
		#7. How far are we from the food in Y?
		#8. Mean of the distance head-body ?
		headS = self.nextHead('STRAIGHT')
		headR = self.nextHead('RIGHT')
		headL = self.nextHead('LEFT')

		input1 = 1 if headS in self.snake or self.size[0] == headS[0] or headS[0] < 0 \
		or self.size[1] == headS[1] or headS[1] < 0 else 0
		input2 = 1 if headL in self.snake or self.size[0] == headL[0] or headL[0] < 0 \
		or self.size[1] == headL[1] or headL[1] < 0 else 0
		input3 = 1 if headR in self.snake or self.size[0] == headR[0] or headR[0] < 0 \
		or self.size[1] == headR[1] or headR[1] < 0 else 0
		
		input4 = 1 if headS == self.food else 0
		input5 = 1 if headL == self.food else 0
		input6 = 1 if headR == self.food else 0

		input7 = abs(self.food[0]-self.snake[-1][0])/self.size[0]
		input8 = abs(self.food[1]-self.snake[-1][1])/self.size[1]
		input9 = mean(list(map(lambda x: distance(self.snake[-1], x), self.snake)))

		return np.array([input1, input2, input3, input4, input5, input6, input7, input8, input9])


	def __str__(self):
		outString = None
		headSign = None
		if self.direction == 'UP':
			headSign = '^'
		elif self.direction == 'DOWN':
			headSign = 'v'
		elif self.direction == 'RIGHT':
			headSign = '>'
		elif self.direction == 'LEFT':
			headSign = '<'

		M = [[' ' for _ in range(self.size[1])] for _ in range(self.size[0])]
		print(self.snake)
		for x,y in self.snake:
			M[x][y] = '#'
		x,y = self.snake[-1]
		M[x][y] = headSign
		x,y = self.food
		M[x][y] = '*'

		outString = '-'*(len(M[0])+2)+'\n'
		for L in M:
			outString += '|'+''.join(L)+'|\n'
		outString += '-'*(len(M[0])+2)+'\n'

		return outString

	__repr__ = __str__


#--------------------------------------

class NeuralNet:
	def __init__(self, inputSize = 9, weights1 = np.array([]), weights2 = np.array([]), \
				weights3 = np.array([]), NpL1 = 20, NpL2 = 20, outN = 3):
		#NpL: Neurons per layer
		#outN: Number of classes for the output
		self.inputSize = inputSize
		self.weights1 = weights1 if len(weights1) else np.random.rand(self.inputSize, NpL1) #pre-layer1
		self.weights2 = weights2 if len(weights2) else np.random.rand(NpL1,NpL2)#post-layer1
		self.weights3 = weights3 if len(weights3) else np.random.rand(NpL2,outN)#post-layer2
		self.NpL1 = NpL1
		self.NpL2 = NpL2
		self.outN = outN
    
	def feedforward(self, x):
		#print(self.inputSize, self.weights1, self.weights2, self.NpL, self.outN)
		layer1 = sigmoid(np.dot(x, self.weights1))
		layer2 = sigmoid(np.dot(layer1, self.weights2))
		output = sigmoid(np.dot(layer2, self.weights3))
		y = softmax(output)
		index = np.where(y == np.max(y))[0][0]
		if index == 0:
			return 'STRAIGHT'
		elif index == 1:
			return 'LEFT'
		elif index == 2:
			return 'RIGHT'
	
	def copy(self):
		return NeuralNet(self.inputSize, self.weights1, self.weights2, self.weights3, self.NpL1, self.NpL2, self.outN)

#-------------------------------------

def geneticAlgorithm(NNsave, N, generation, meanOn = 20, keepBest = 0.1, mutate = 0.4, merge = 0.3):
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
		rewards1 = []
		for _ in range(meanOn):
			NNList1 = [NeuralNet() for _ in range(N)]
			games1 = [SnakeGame() for _ in range(N)]
			for k in range(N):
				games1[k].NNetPlay(NNList1[k])
				if len(rewards1) != N:
					rewards1.append((NNList1[k], games1[k].reward*1/meanOn))
				else:
					rewards1[k] = (rewards1[k][0], rewards1[k][1] + rewards1[k][1]*1/meanOn)

		NNsave[1] = sorted(rewards1, key=lambda x: x[1], reverse=True)#Sort by reward
		keys = [1]

	#Check if we need to compute something or if all gen are already calculated
	maxGen = max(keys)
	if maxGen >= generation:
		return

	lastGen = NNsave[maxGen]
	for genNb in tqdm(range(maxGen, generation)):
		keptL = NNsave[genNb][1:int(N*keepBest)]
		keptL = [x[0] for x in keptL]#take only the NN

		mutateL = keptL[:int(N*mutate)]
		#mutateL = [x[0] for x in mutateL]

		mergeL = keptL[int(N*mutate):int(N*(mutate+merge))]
		#mergeL = [x[0] for x in mergeL]

		newGen = keptL.copy()
		#----------- mutation
		for NN in mutateL:
			NNcpy = NN.copy()
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

		#----------- reproduction
		lastNN = lastGen[0][0]
		for NN in mergeL:
			NNcpy == NN.copy()
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

		#------------ evaluation
		rewards = []
		for _ in range(meanOn):
			games = [SnakeGame() for _ in range(N)]
			for k in range(N):
				games[k].NNetPlay(newGen[k])
				if len(rewards) != N:
					rewards.append((newGen[k], games[k].reward*1/meanOn))
				else:
					rewards[k] = (rewards[k][0], rewards[k][1]+rewards[k][1]*1/meanOn)

		NNsave[genNb+1] = sorted(rewards, key=lambda x: x[1], reverse=True) #Sort by reward

	return

def showEvolution(NNsave):
	rewardL = []
	for k in list(NNsave.keys()):
		G = SnakeGame()
		G.NNetPlayShow(NNsave[k][0][0], generation = k)
		rewardL.append(G.reward)
	for k in list(NNsave.keys()):
		print('Generation {} - reward {}'.format(k,rewardL[k-1]))




#G = SnakeGame()
#G.hummanPlay()

# Nrand = NeuralNet()
# G.NNetPlayShow(Nrand)



try:
	NNsave = pickle.load(open( "NNsave.pickle", "rb" ))
except:
	NNsave = {}

geneticAlgorithm(NNsave, 1000, 15, meanOn = 10)
showEvolution(NNsave)

# NN1 = NNsave[9][0][0]
# NN2 = NNsave[10][0][0]
# rw1 = 0
# rw2 = 0
# N = 100
# for k in range(N):
# 	G1 = SnakeGame()
# 	G1.NNetPlay(NN1)
# 	rw1 += G1.reward*1/N
# 	G2 = SnakeGame()
# 	G2.NNetPlay(NN2)
# 	rw2 += G2.reward*1/N
# print(rw1, rw2)

pickle.dump(NNsave, open("NNsave.pickle", "wb"))

