from TestGame import Test
from Agent.RandomAgent import RandomAgent
ra = RandomAgent(4)
test = Test('PP4a')
returns = test.test_game(10, ra)
print(returns)