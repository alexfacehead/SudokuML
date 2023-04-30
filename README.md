# SudokuML

* A simple Q-learning (reinforcement learning) based agent that solves sudoku puzzles.

# Features

* Very clear structure and function definitions with lots of hints for you to use
* Completely modular design
* Detailed debug output
* Can switch out the model design for anything you like

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

The script will execute the Sudoku reinforcement learning program using the specified configurations in the main code. It will read the `.env` file to determine the appropriate file paths for Google Colab or local environments.


# To Do

* Train a great model!
* Implement grid-search for hyper parameters
