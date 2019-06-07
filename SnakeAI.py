import random as rd
import collections as c
import curses
import time
import numpy as np

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def softmax(A):  
    expA = np.exp(A)
    return expA / expA.sum()

def distance(x, y):
	return np.linalg.norm(np.array(x)-np.array(y))

#With size 10,10 the position allowed are (0,0) and (9,9)
#self.snake : Queue([tail..... head])
class SnakeGame:
	SPEED = 600

	def __init__(self, size = (10,20), maxMoves=400):
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

		if head in self.snake or self.size[0] == head[0] or head[0] < 0 or self.size[1] == head[1] or head[1] < 0:
			return False, self.score

		elif head == self.food: # we move to the food
			self.score += 1
			self.reward += 10
			self.snake.append(head)
			self.chooseFoodPosition()
			self.remainingMoves = self.maxMoves
		else: # we move but no food eaten
			if distance(self.snake[-1], self.food) <= distance(head, self.food): 
			# we are further from the food
				self.reward -= 1.3
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

	def createInputNN(self):
		#Input:
		#1. Is it clear STRAIGHT ?
		#2. Is it clear LEFT ?
		#3. Is it clear RIGHT ?
		#4. Is there food STRAIGHT ?
		#5. Is there food LEFT ?
		#6. Is there food RIGHT ?
		headS = self.nextHead('STRAIGHT')
		headR = self.nextHead('RIGHT')
		headL = self.nextHead('LEFT')

		input1 = 1 if headS in self.snake or self.size[0] == headS[0] or headS[0] < 0 \
		or self.size[1] == headS[1] or headS[1] < 0 else 0
		input2 = 1 if headL in self.snake or self.size[0] == headL[0] or headL[0] < 0 \
		or self.size[1] == headL[1] or headL[1] < 0 else 0
		input3 = 1 if headR in self.snake or self.size[0] == headR[0] or headR[0] < 0 \
		or self.size[1] == headR[1] or headR[1] < 0 else 0
		
		input4 = 
		input5 = 
		input6 = 


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
	#Input:
	#1. Is it clear STRAIGHT ?
	#2. Is it clear LEFT ?
	#3. Is it clear RIGHT ?
	#4. Is there food STRAIGHT ?
	#5. Is there food LEFT ?
	#6. Is there food RIGHT ?

	def __init__(self, x, NpL = 15, outN = 3):
		#NpL: Neurons per layer
		#outN: Number of classes for the output
		self.input = x
		self.weights1 = np.random.rand(self.input.shape[0],NpL) #pre-layer1
		self.weights2 = np.random.rand(NpL,outN)#post-layer1
		self.output = np.zeros(self.weights2.shape[1])
    
	def feedforward(self):
		self.layer1 = sigmoid(np.dot(self.input, self.weights1))
		print(self.input, self.weights1, self.layer1)
		self.output = sigmoid(np.dot(self.layer1, self.weights2))
		print(softmax(self.output))
		return softmax(self.output)


G = SnakeGame()

N = NeuralNet(np.array([1,0,1]))
N.feedforward()

