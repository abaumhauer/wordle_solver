# Wordle Solver 🦀

An ultra-fast, information-theoretic Wordle solver written in Rust. It uses [Shannon Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) to calculate the mathematically optimal guess at every turn.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)

## Features

* **Entropy-based Logic:** Calculates the Information Gain (in bits) of every possible word to find the guess that eliminates the most possibilities.
* **Parallel Processing:** Leverages `rayon` to utilize all CPU cores, checking millions of game states in milliseconds.
* **Smart Caching:** Automatically pre-computes and caches the expensive "First Turn" logic. Startup is instant after the first run.
* **Flexible Input:** Supports both **Mask Mode** (color codes) and **Letter Mode** (case-sensitive re-typing).
* **Hardened:** Robust input validation and corruption resilience.
* **Variable Lengths:** Supports 3, 4, 5, 6+ letter dictionaries automatically.

## Installation

### Prerequisites
You need [Rust and Cargo](https://rustup.rs/) installed (Edition 2024 or newer).

### Build from Source
```bash
git clone [https://github.com/abaumhauer/wordle_solver.git](https://github.com/abaumhauer/wordle_solver.git)
cd wordle_solver
cargo build --release
```

The executable will be located in `./target/release/wordle_solver`.

## Usage

### 1. Basic Run
Ensure you have a dictionary file (e.g., `words.txt`) in the same directory.
```bash
./target/release/wordle_solver
```

### 2. Command Line Options
```bash
# Use a custom dictionary path
./wordle_solver --word-list my_words.txt

# Force a cache rebuild (ignore existing .cache file)
./wordle_solver --force-rebuild

# Pre-calculate cache and exit (useful for build scripts/CI)
./wordle_solver --calc-only
```

### 3. How to Play
1.  The solver suggests a word (e.g., `TRACE`).
2.  Play that word in your actual Wordle game.
3.  Enter the feedback into the solver using one of two modes:

    **Mode A: Mask Code (Faster)**
    * `g` = Green (Correct)
    * `y` = Yellow (Wrong Position)
    * `-` = Gray (Not in word)
    * *Example:* `g y - - g`

    **Mode B: Letter Case**
    * **UPPER CASE** = Green
    * **lower case** = Yellow
    * **Symbol (- or .)** = Gray
    * *Example:* `T r - - E`

## Performance
On a modern multi-core CPU (Apple M-series or Intel i7+), the solver calculates the optimal opener for a ~12,000 word dictionary in under **1 second** (uncached) or **<100ms** (cached).

## License
MIT
