from TestGame import Test
from Agent.RandomAgent import RandomAgent
ra = RandomAgent(4)
test = Test('PP4a')
returns = test.test_game(20, ra, 20)
print(returns)