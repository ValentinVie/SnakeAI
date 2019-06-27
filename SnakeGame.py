import random as rd
import collections as c
import curses
import time
import numpy as np



#---------------- MISC

def distance(x, y):
	return np.linalg.norm(np.array(x)-np.array(y))

def mean(L):
	return sum(L)/len(L)


#---------------- SnakeGame
class SnakeGame:
	SPEED = 600
	SPEEDNN = 100

	def __init__(self, size = (10,20), trainingGame = True, maxMoves=100):
		#With size 10,10 the position allowed are (0,0) and (9,9)
		#self.snake : Queue([tail..... head])

		self.size = size
		self.maxMoves = maxMoves
		self.remainingMoves = self.maxMoves
		self.snake = c.deque()
		self.snake.append((int(size[0]/2), int(size[1]/2)))
		self.direction = 'DOWN'
		self.score = 0
		self.reward = 0
		if trainingGame: 
			# It is a training game we only train over 10 games.
			#A game is a unique succession of food placement.
			#We basically bias the random number generator to only have 10 sequences
			self.randgen = rd.Random(rd.randint(0,100))
		else: 
			#The game has not been seen before. The random number 
			#generator generate a sequence of food placement never seen before.
			self.randgen = rd.Random(rd.randint(10,10000))

		self.chooseFoodPosition()		

	def chooseFoodPosition(self):
		if self.score == self.size[0]*self.size[1]:
			raise NameError('Food position could not be determined.')
			return False
		self.food = ((self.randgen.randint(1,self.size[0])-1, self.randgen.randint(1,self.size[1])-1))
		while self.food in self.snake:
			self.food = ((self.randgen.randint(1,self.size[0])-1, self.randgen.randint(1,self.size[1])-1))
		return True

	def nextHead(self, nextMove, directionChange = 0):
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
			if directionChange:
				self.direction = directionL[(directionL.index(self.direction)+1)%4]
		elif nextMove == 'LEFT':
			dx,dy = movementsTurns[self.direction]
			head = (head[0]-dx, head[1]-dy)
			if directionChange:
				self.direction = directionL[(directionL.index(self.direction)-1)%4]
		elif nextMove == 'STRAIGHT':
			dx,dy = movementsStraight[self.direction]
			head = (head[0]+dx, head[1]+dy)

		return head

	def positionInWall(self, pos):
		return self.size[0] == pos[0] or pos[0] < 0 or self.size[1] == pos[1] or pos[1] < 0

	def nextMoveUpdate(self, nextMove):
		nextHead = self.nextHead(nextMove, directionChange = 1)

		if nextHead in self.snake or self.positionInWall(nextHead):
			# if not headSGameOver or not headRGameOver or not headLGameOver:
			# 	self.reward -= 200
			return False

		elif nextHead == self.food: # we move to the food
			self.score += 1
			self.reward += 20
			self.snake.append(nextHead)
			self.chooseFoodPosition()
			self.remainingMoves = self.maxMoves
		else: # we move but no food eaten
			if distance(self.snake[-1], self.food) <= distance(nextHead, self.food): 
			# we are further from the food
				self.reward -= 1.5
			else:
			# we got closer to the food
				self.reward += 1

			self.snake.append(nextHead)
			self.snake.popleft()
			self.remainingMoves -= 1

		if self.remainingMoves == 0:
			return False

		return True

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
					gameOn = self.nextMoveUpdate('RIGHT')
				elif char == curses.KEY_LEFT:
					gameOn = self.nextMoveUpdate('LEFT')
				else:
					gameOn = self.nextMoveUpdate('STRAIGHT')

				if not gameOn:
					break
				
				#Remove caracters on screen
				if prevSnake[0] != self.snake[0]:
					screen.addch(prevSnake[0][0]+1, prevSnake[0][1]+1, ' ')
				prevSnake = self.snake.copy()

				if prevFood != self.food:
					screen.addch(prevFood[0]+1, prevFood[1]+1, ' ')
					prevFood = self.food
				
				#Add caracters on screen
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
			gameOn = self.nextMoveUpdate(model.feedforward(self.createInputNN()))

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

				gameOn = self.nextMoveUpdate(model.feedforward(self.createInputNN()))

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
		headS = self.nextHead('STRAIGHT')
		headL = self.nextHead('LEFT')
		headR = self.nextHead('RIGHT')
		head = self.snake[-1]

		input1 = 1 if headS in self.snake or self.positionInWall(headS) else 0
		input2 = 1 if headL in self.snake or self.positionInWall(headL) else 0
		input3 = 1 if headR in self.snake or self.positionInWall(headR) else 0
		
		if self.direction == 'RIGHT':
			input4 = 1 if head[0] == self.food[0] and head[1] <= self.food[1] else 0
			input5 = 1 if head[1] == self.food[1] and head[0] >= self.food[0] else 0
			input6 = 1 if head[1] == self.food[1] and head[0] <= self.food[0] else 0
		elif self.direction == 'LEFT':
			input4 = 1 if head[0] == self.food[0] and head[1] >= self.food[1] else 0
			input5 = 1 if head[1] == self.food[1] and head[0] <= self.food[0] else 0
			input6 = 1 if head[1] == self.food[1] and head[0] >= self.food[0] else 0
		elif self.direction == 'DOWN':
			input4 = 1 if head[1] == self.food[1] and head[0] <= self.food[0] else 0
			input5 = 1 if head[0] == self.food[0] and head[1] <= self.food[1] else 0
			input6 = 1 if head[0] == self.food[0] and head[1] >= self.food[1] else 0
		elif self.direction == 'UP':
			input4 = 1 if head[1] == self.food[1] and head[0] >= self.food[0] else 0
			input5 = 1 if head[0] == self.food[0] and head[1] >= self.food[1] else 0
			input6 = 1 if head[0] == self.food[0] and head[1] <= self.food[1] else 0

		return np.array([input1, input2, input3, input4, input5, input6])


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


if __name__ == "__main__": #Testing
	G = SnakeGame()
	G.hummanPlay()

	from NeuralNet import NeuralNet

	G = SnakeGame()
	NNRandom = NeuralNet()
	G.NNetPlayShow(NNRandom)


