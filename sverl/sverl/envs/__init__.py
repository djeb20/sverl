from gymnasium.envs.registration import register

register(
    id='Mastermind-v0',
    entry_point='sverl.envs.mastermind:Mastermind',
)

register(
    id='FactoredTaxi-v3',
    entry_point='sverl.envs.taxi:FactoredTaxi',
)

register(
    id='Grid-v0',
    entry_point='sverl.envs.grid:Grid',
)

register(
    id='ColourGrid-v0',
    entry_point='sverl.envs.colour_grid:ColourGrid',
)

register(
    id='Dice-v0',
    entry_point='sverl.envs.dice:Dice',
)

register(
    id='MiniMinesweeper-v0',
    entry_point='sverl.envs.mini_minesweeper:MiniMinesweeper',
)

register(
    id='TicTacToe-v0',
    entry_point='sverl.envs.tic_tac_toe:TicTacToe',
)