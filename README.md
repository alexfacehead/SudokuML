# SudokuML

* A simple Q-learning (reinforcement learning) based agent that solves sudoku puzzles.

# Features

* Very clear structure and function definitions with lots of hints for you to use
* Completely modular design
* Detailed debug output
* Can switch out the model design for anything you like
* Evaluation!
* Grid search over whatever seach space you want for hyperparameters, so you don't have to decide =)

# Usage

1. Install the required dependencies:

`
pip install -r requirements.txt
`

2. Create a `.env` file in the project directory and add the following variables:

`
google_colab_path=/content/drive/MyDrive/SudokuML/weights
local_path=/home/dev/SudokuML/weights
`

Replace the paths with the appropriate paths for your Google Colab and local environments. 
If you are only using one or the other, that's ok.

3. Run the main script:

* If you're running the script for the first time or want to delete the existing `debug_output.txt`, use the `--fresh` flag:

  `
  python Main.py --fresh
  `

* If you want to continue using the existing `debug_output.txt` without deleting it, run the script without the `--fresh` flag:

  `
  python Main.py
  `

  You can also use the `--force` flag to force an evaluation at the very start so you can see how well it does on 5 tests of whatever DataLoader object you choose to run it on.

The script will execute the Sudoku reinforcement learning program using the specified configurations in the main code. It will read the `.env` file to determine the appropriate file paths for Google Colab or local environments.

# Training Guide
* It's probably best to start your model out with a low-difficulty puzzle
* To make a puzzle, use the `DataLoader` object. Difficulty is a simple `int` between `1-18` and represents the number of zeros that your agent will have to fill in. 
* Use the resources folder, or create your own dataset using the `kaggle` sudoku dataset: https://www.kaggle.com/datasets/bryanpark/sudoku
* If you choose to use the resources that are already here, you will have `sudoku_mini_easy.csv`, `sudoku_mini_medium.csv` and `sudoku_mini_small.csv`. They are all similar, 100 puzzle-sized datasets.
* Use `data_loader = DataLoader("./resources/sudoku_mini_medium.csv", 24)` and make sure the input file and difficulty are as you expect.
* Pass that into your agent, and let the agent run for an epoch to produce weights.

Enjoy your agent! Evaluate it afterward if you feel like it: the training loop always loads the *most recent weights*.

# To Do

* Train a great model!
* Implement grid-search for hyper parameters
