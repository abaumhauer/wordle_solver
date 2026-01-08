use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;

// -----------------------------------------------------------------------------
// EXTERNAL CRATES
// -----------------------------------------------------------------------------
// bincode: For fast, compact binary serialization of the cache file.
// clap:    For parsing command line arguments and generating help menus.
// rayon:   For parallel processing of the entropy calculation (speed).
// serde:   For serializing our data structures to disk.
// sha2:    For detecting if the dictionary file has changed (invalidating cache).
use bincode::Options;
use clap::Parser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// =============================================================================
// 1. DATA STRUCTURES & CLI CONFIGURATION
// =============================================================================

/// An optimized, Information-Theoretic Wordle Solver.
///
/// This program calculates the "Shannon Entropy" of every possible word to
/// determine which guess provides the most information (eliminates the most
/// wrong answers).
///
/// It supports:
/// - Custom dictionaries (words.txt)
/// - Variable word lengths (3 to 8+ letters)
/// - Caching of expensive initial calculations
/// - Parallel processing for speed
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the dictionary file containing valid words (one per line).
    /// The solver automatically detects the word length from the first valid line.
    #[arg(short, long, default_value = "words.txt")]
    word_list: String,

    /// Force a rebuild of the optimization cache.
    /// Use this if you suspect the cache file (.cache) is corrupted or stale,
    /// though the solver usually detects this automatically via hashing.
    #[arg(short, long, default_value_t = false)]
    force_rebuild: bool,

    /// Run the initial optimization, dump the cache file, and exit immediately.
    /// Useful for pre-computing heavy data on a build server.
    #[arg(long, default_value_t = false)]
    calc_only: bool,
}

/// Represents the specific feedback Wordle gives for a single letter.
/// We derive Copy/Clone because this enum is tiny (1 byte) and cheap to pass around.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum Feedback {
    Green,  // Correct letter, Correct position.
    Yellow, // Correct letter, Wrong position.
    Gray,   // Letter not in word (or is an excess duplicate).
}

/// We use `Vec<char>` instead of `String` for the "Word" type.
/// `String` in Rust is UTF-8 encoded, which makes indexing (word[i]) O(N).
/// `Vec<char>` allows O(1) random access, which is critical for the millions of
/// tight-loop comparisons we perform in the logic engine.
type Word = Vec<char>;

/// The schema for the binary cache file stored on disk.
/// This allows us to skip the expensive "First Turn" calculation on subsequent runs.
#[derive(Serialize, Deserialize, Debug)]
struct CacheFile {
    /// SHA256 hash of the source text file.
    /// If the user edits `words.txt`, this hash changes, invalidating the cache.
    source_hash: String,

    /// The length of words in this cache (e.g., 5).
    /// Prevents loading a 5-letter cache for a 6-letter dictionary.
    word_length: usize,

    /// The pre-computed list of best starting words, sorted by entropy.
    suggestions: Vec<(String, f64)>,
}

// =============================================================================
// 2. CORE LOGIC ENGINE (THE MATH)
// =============================================================================

/// Determines the coloring (Feedback) for a `guess` against a specific `solution`.
///
/// # The Algorithm
/// Wordle has strict rules for double letters that this function must handle:
/// 1. **Greens take priority:** We mark all exact matches first. They "consume"
///    that letter from the solution's available pool.
/// 2. **Yellows check availability:** A letter is marked Yellow only if it exists
///    in the solution AND hasn't already been "consumed" by a Green match or
///    a previous Yellow match (scanning left-to-right).
fn calculate_feedback(guess: &[char], solution: &[char]) -> Vec<Feedback> {
    let len = guess.len();
    let mut feedback = vec![Feedback::Gray; len];
    let mut solution_counts = HashMap::new();

    // Step A: Build frequency map of the solution (e.g., "EERIE" -> E:3, R:1, I:1)
    for &c in solution {
        *solution_counts.entry(c).or_insert(0) += 1;
    }

    // Step B: Pass 1 - Identify GREENS (Correct Position)
    // We must do this first so they claim "ownership" of the letters.
    for i in 0..len {
        if guess[i] == solution[i] {
            feedback[i] = Feedback::Green;
            if let Some(count) = solution_counts.get_mut(&guess[i]) {
                *count -= 1;
            }
        }
    }

    // Step C: Pass 2 - Identify YELLOWS (Wrong Position)
    for i in 0..len {
        // Skip if already marked Green
        if feedback[i] == Feedback::Green {
            continue;
        }

        let letter = guess[i];
        // We only mark Yellow if the letter is still available in our counts
        if let Some(count) = solution_counts.get_mut(&letter)
            && *count > 0
        {
            feedback[i] = Feedback::Yellow;
            *count -= 1;
        }
    }

    feedback
}

/// Calculates the Shannon Entropy (Information Gain) of a specific `guess`.
///
/// # Logic
/// 1. We assume `guess` is played against *every* remaining `possible_solution`.
/// 2. We count how many times each Feedback Pattern (e.g., Green-Gray-Yellow...) occurs.
/// 3. A guess that produces many different patterns (splitting the solutions into small groups)
///    has High Entropy. A guess that produces the same pattern for everyone has Low Entropy.
///
/// # Returns
/// A float representing bits of information. Higher is better.
#[allow(clippy::cast_precision_loss)]
fn calculate_entropy(guess: &[char], possible_solutions: &[Word]) -> f64 {
    let mut pattern_counts: HashMap<Vec<Feedback>, usize> = HashMap::new();

    // Simulate the guess against all possibilities
    for solution in possible_solutions {
        let pattern = calculate_feedback(guess, solution);
        *pattern_counts.entry(pattern).or_insert(0) += 1;
    }

    let total_solutions = possible_solutions.len() as f64;
    let mut entropy = 0.0;

    // Shannon Entropy Formula: H = - Sum(p(x) * log2(p(x)))
    // Where p(x) is the probability of getting a specific feedback pattern.
    for &count in pattern_counts.values() {
        let p = count as f64 / total_solutions;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    entropy
}

/// Finds the best words to guess next by calculating entropy for all candidates.
///
/// # Parallelization
/// This function uses `rayon` (`par_iter`) to utilize all CPU cores.
/// Calculating entropy is O(N * M) where N=Candidates and M=Solutions.
/// For a full dictionary (12k words), this is computationally heavy.
fn find_best_guess(candidates: &[Word], possible_solutions: &[Word]) -> Vec<(String, f64)> {
    // Optimization: If we are down to 1 or 2 words, entropy is overkill.
    // Just guessing one of the remaining words guarantees a win in 1 or 2 turns.
    if possible_solutions.len() <= 2 {
        return possible_solutions
            .iter()
            .map(|w| (w.iter().collect(), 0.0))
            .collect();
    }

    // Parallel map-reduce logic
    let mut scores: Vec<(String, f64)> = candidates
        .par_iter()
        .map(|guess| {
            let score = calculate_entropy(guess, possible_solutions);
            (guess.iter().collect::<String>(), score)
        })
        .collect();

    // Sort descending (Highest Entropy first)
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    scores
}

// =============================================================================
// 3. CACHING & FILE I/O
// =============================================================================

/// Computes a unique SHA256 hash of the dictionary file contents.
/// Used to verify if the cache is still valid or if the user edited words.txt.
fn compute_file_hash(path: &str) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;
    Ok(hex::encode(hasher.finalize()))
}

/// Loads the raw dictionary file.
///
/// - Scans the file line by line.
/// - Automatically detects word length from the first valid word found.
/// - Filters out garbage (numbers, symbols, empty lines).
/// - Enforces uniform word length (discards words that don't match the detected length).
fn load_raw_dictionary(path: &str) -> io::Result<(Vec<Word>, usize)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut words = Vec::new();
    let mut target_length: Option<usize> = None;

    for line_res in reader.lines() {
        let line = line_res?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Hardening: Ensure only alphabetic characters
        if !trimmed.chars().all(char::is_alphabetic) {
            continue;
        }

        let chars: Vec<char> = trimmed.to_uppercase().chars().collect();

        match target_length {
            Some(len) => {
                if chars.len() != len {
                    // Skip mixed length words to maintain game logic integrity
                    continue;
                }
            }
            None => {
                // First valid word establishes the rules (length) for this game
                target_length = Some(chars.len());
            }
        }

        words.push(chars);
    }

    if words.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No valid words found in file.",
        ));
    }

    Ok((words, target_length.unwrap()))
}

/// Returns a hardened Bincode configuration.
/// Sets a memory limit (64MB) to prevent "zip bomb" style crashes if a cache file is corrupted
/// and the length prefix interprets as an Exabyte size.
fn get_bincode_config() -> impl bincode::Options {
    bincode::options()
        .with_limit(64 * 1024 * 1024)
        .with_little_endian()
        .with_fixint_encoding()
}

/// The Smart Loader logic.
/// 1. Checks if a valid cache file exists (hash match + length match).
/// 2. If valid, loads suggestions instantly (0.1s).
/// 3. If invalid (or missing), runs the heavy calculation (5-30s) and saves a new cache.
fn get_initial_suggestions(
    dict_path: &str,
    all_words: &[Word],
    force_rebuild: bool,
) -> Vec<(String, f64)> {
    let cache_path = PathBuf::from(format!("{dict_path}.cache"));
    let current_hash = compute_file_hash(dict_path).unwrap_or_else(|_| "unknown".to_string());
    let word_len = all_words[0].len();

    // Attempt Cache Load
    if !force_rebuild
        && cache_path.exists()
        && let Ok(file) = File::open(&cache_path)
    {
        let reader = BufReader::new(file);
        // We construct the config fresh to avoid ownership moves
        match get_bincode_config().deserialize_from::<_, CacheFile>(reader) {
            Ok(cache) => {
                if cache.source_hash == current_hash && cache.word_length == word_len {
                    println!("Loaded optimized opening words from cache.");
                    return cache.suggestions;
                }
                println!("Cache outdated (hash/length mismatch). Recalculating...");
            }
            Err(_) => {
                println!("Cache file corrupted or incompatible. Recalculating...");
            }
        }
    }

    // Cache Miss: Calculate
    println!("Calculating initial entropy (this takes time for new dictionaries)...");
    let suggestions = find_best_guess(all_words, all_words);

    // Save New Cache
    let cache_data = CacheFile {
        source_hash: current_hash,
        word_length: word_len,
        suggestions: suggestions.clone(),
    };

    if let Ok(file) = File::create(&cache_path) {
        if let Err(e) = get_bincode_config().serialize_into(file, &cache_data) {
            eprintln!("Warning: Failed to write cache file: {e}");
        } else {
            println!("Saved cache to {}", cache_path.display());
        }
    }

    suggestions
}

// =============================================================================
// 4. INPUT PARSING & VALIDATION
// =============================================================================

/// Parses raw string input into Feedback.
///
/// This uses a unified parser that detects modes heuristically to avoid ambiguity.
///
/// # Ambiguity Handling (The "BUGGY" Problem)
/// If a guess is "BUGGY" and the user types `g`, did they mean "Green" or the letter 'g'?
/// - If the input is **entirely** valid color codes (g, y, -), we assume **Mask Mode**.
/// - Otherwise, we assume **Letter Mode**.
fn parse_feedback(
    input: &str,
    current_guess_str: &str,
    target_len: usize,
) -> Result<Vec<Feedback>, String> {
    // Clean input of whitespace
    let cleaned: String = input.chars().filter(|c| !c.is_whitespace()).collect();

    if cleaned.len() != target_len {
        return Err(format!(
            "Input must be {target_len} characters. You entered {}.",
            cleaned.len()
        ));
    }

    let input_chars: Vec<char> = cleaned.chars().collect();
    let guess_chars: Vec<char> = current_guess_str.chars().collect();
    let mut feedback = vec![Feedback::Gray; target_len];

    // Heuristic: Mask Mode if all chars are valid mask symbols.
    // This solves edge cases like "BUGGY" where 'g' and 'y' are letters in the word.
    let is_mask_mode = input_chars.iter().all(|c| {
        let lower = c.to_ascii_lowercase();
        lower == 'g' || lower == 'y' || "-._".contains(lower)
    });

    if is_mask_mode {
        // --- MODE A: Mask Code (g/y/-) ---
        for i in 0..target_len {
            match input_chars[i].to_ascii_lowercase() {
                'g' => feedback[i] = Feedback::Green,
                'y' => feedback[i] = Feedback::Yellow,
                '-' | '.' | '_' => feedback[i] = Feedback::Gray,
                _ => unreachable!(), // Heuristic guarantees these are valid
            }
        }
    } else {
        // --- MODE B: Letter Re-entry ---
        for i in 0..target_len {
            let c = input_chars[i];
            let guess_c = guess_chars[i];

            if "-._".contains(c) {
                // Explicit Gray symbol
                feedback[i] = Feedback::Gray;
            } else if c.eq_ignore_ascii_case(&guess_c) {
                // Letter Match Logic
                if c.is_uppercase() {
                    feedback[i] = Feedback::Green;
                } else {
                    // Lowercase match = Yellow.
                    // If user meant Gray, they should have used '-'
                    feedback[i] = Feedback::Yellow;
                }
            } else {
                return Err(format!(
                    "Invalid input '{c}'. It is not a color code (g/y/-) \
                     and does not match the guess letter '{guess_c}'."
                ));
            }
        }
    }

    Ok(feedback)
}

/// Validates that a manual user guess matches the game constraints (length, alpha).
fn sanitize_user_guess(raw: &str, target_len: usize) -> Option<Word> {
    let trimmed = raw.trim();
    if trimmed.len() != target_len {
        return None;
    }
    if !trimmed.chars().all(char::is_alphabetic) {
        return None;
    }
    Some(trimmed.to_uppercase().chars().collect())
}

/// Helper to print the grid of remaining possibilities neatly.
fn print_candidates(words: &[Word], word_len: usize) {
    let term_width = term_size::dimensions().map_or(80, |(w, _)| w);
    let col_width = word_len + 2;
    let cols = term_width / col_width;

    println!("Possible Solutions ({}):", words.len());
    println!("{:-<1$}", "", term_width);

    for (i, word) in words.iter().enumerate() {
        let w_str: String = word.iter().collect();
        print!("{w_str: <col_width$}");
        if (i + 1) % cols == 0 {
            println!();
        }
    }
    println!("\n{:-<1$}", "", term_width);
}

// =============================================================================
// 5. MAIN APPLICATION LOOP
// =============================================================================

#[allow(clippy::too_many_lines)]
fn main() -> io::Result<()> {
    // 1. Parse CLI Args
    let args = Args::parse();

    // 2. Load Dictionary
    println!("Loading dictionary from '{}'...", args.word_list);
    let (all_words, word_len) = match load_raw_dictionary(&args.word_list) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Error loading dictionary: {e}");
            eprintln!("Ensure the file exists and contains valid alphabetical words.");
            return Ok(());
        }
    };

    println!("Dictionary loaded. Word length detected: {word_len}");

    // 3. Load or Calculate Starting Strategy
    let initial_suggestions =
        get_initial_suggestions(&args.word_list, &all_words, args.force_rebuild);

    // If --calc-only flag was set, we stop here.
    if args.calc_only {
        println!("Calculation complete. Cache updated.");
        println!(
            "Top Opener: {} (Entropy: {:.2})",
            initial_suggestions[0].0, initial_suggestions[0].1
        );
        return Ok(());
    }

    // 4. Initialize Game State
    let mut possible_solutions = all_words.clone();
    let dictionary_set: HashSet<Word> = all_words.iter().cloned().collect();

    // 5. Print Instructions
    println!("------------------------------------------------------------");
    println!("HOW TO PLAY:");
    println!("1. The solver suggests the statistically optimal word.");
    println!("2. Press [ENTER] to play that word, or type your own.");
    println!("3. Enter the result from the game using one of two formats:");
    println!();
    println!("   MODE A: Mask Code (Best for speed)");
    println!("   Type colors directly: 'g' (Green), 'y' (Yellow), '-' (Gray)");
    println!("   Example: g y - - g");
    println!();
    println!("   MODE B: Letter Re-entry");
    println!("   Re-type the letters that match, use symbols for misses.");
    println!("   - UPPER CASE letter = Green (Correct Position)");
    println!("   - lower case letter = Yellow (Wrong Position)");
    println!("   - Dash '-' or Dot '.' = Gray (Not in word)");
    println!("   Example: T r - - E");
    println!("------------------------------------------------------------");

    let mut current_suggestions = initial_suggestions;

    // 6. Game Loop (Max 6 turns)
    for turn in 1..=6 {
        println!("\n=== TURN {turn} ===");

        // Display remaining candidates (if list is short)
        if possible_solutions.len() < 100 {
            print_candidates(&possible_solutions, word_len);
        } else {
            println!("{} possible solutions remaining.", possible_solutions.len());
        }

        if possible_solutions.len() == 1 {
            println!(
                "The word is: {}",
                possible_solutions[0].iter().collect::<String>()
            );
            break;
        }

        // Recalculate Entropy (if past turn 1)
        if turn > 1 {
            println!("Thinking...");
            current_suggestions = find_best_guess(&all_words, &possible_solutions);
        }

        if current_suggestions.is_empty() {
            println!("No valid words remaining.");
            break;
        }

        let recommended_word = &current_suggestions[0].0;
        println!(
            "Recommended: {} (Entropy: {:.2})",
            recommended_word, current_suggestions[0].1
        );

        // --- INTERACTION PHASE 1: WHAT DID WE PLAY? ---
        let played_word_str = loop {
            print!("Word Played [Enter for '{recommended_word}'] > ");
            io::stdout().flush()?;

            let mut buf = String::new();
            io::stdin().read_line(&mut buf)?;
            let raw = buf.trim();

            if raw.is_empty() {
                break recommended_word.clone();
            }

            if let Some(w) = sanitize_user_guess(raw, word_len) {
                // Warn if user plays a word not in our dictionary, but allow it
                if !dictionary_set.contains(&w) {
                    println!(
                        "Warning: '{}' is not in the dictionary. Proceeding anyway...",
                        raw.to_uppercase()
                    );
                }
                break raw.to_uppercase();
            }

            println!("Invalid word. Must be {word_len} alphabetic letters.");
        };

        // --- INTERACTION PHASE 2: WHAT WAS THE FEEDBACK? ---
        let feedback = loop {
            print!("Result for '{played_word_str}' > ");
            io::stdout().flush()?;

            let mut buf = String::new();
            io::stdin().read_line(&mut buf)?;

            match parse_feedback(&buf, &played_word_str, word_len) {
                Ok(fb) => break fb,
                Err(e) => println!("Error: {e}"),
            }
        };

        // --- FILTER PHASE ---
        let guess_chars: Vec<char> = played_word_str.chars().collect();
        let prev_count = possible_solutions.len();

        // Keep only solutions that would have produced this exact feedback
        possible_solutions.retain(|sol| calculate_feedback(&guess_chars, sol) == feedback);

        println!(
            "Eliminated {} words.",
            prev_count - possible_solutions.len()
        );

        if possible_solutions.is_empty() {
            println!("ERROR: No words match that feedback.");
            break;
        }
    }

    Ok(())
}

// =============================================================================
// 6. TEST HARNESS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    // Helper to make test word construction cleaner
    fn to_word(s: &str) -> Word {
        s.chars().collect()
    }

    // --- A. Logic Tests ---

    #[test]
    fn test_simple_feedback() {
        let guess = to_word("ABCDE");
        let solution = to_word("ABCFG");
        let expected = vec![
            Feedback::Green,
            Feedback::Green,
            Feedback::Green,
            Feedback::Gray,
            Feedback::Gray,
        ];
        assert_eq!(calculate_feedback(&guess, &solution), expected);
    }

    #[test]
    fn test_double_letter_logic() {
        // Solution: ABBEY (Bs at 1, 2)
        // Guess: BABES
        let solution = to_word("ABBEY");
        let guess = to_word("BABES");
        let fb = calculate_feedback(&guess, &solution);

        assert_eq!(fb[2], Feedback::Green); // B matches B
        assert_eq!(fb[3], Feedback::Green); // E matches E
        assert_eq!(fb[0], Feedback::Yellow); // B[0] is in word, wrong spot
        assert_eq!(fb[1], Feedback::Yellow); // A[1] is in word, wrong spot
    }

    #[test]
    fn test_triple_letter_overflow() {
        // Solution has 3 Es. Guess has 3 Es, but at different spots.
        let solution = to_word("EERIE");
        let guess = to_word("GEESE");
        let fb = calculate_feedback(&guess, &solution);

        assert_eq!(fb[1], Feedback::Green);
        assert_eq!(fb[4], Feedback::Green);
        assert_eq!(fb[2], Feedback::Yellow);
        assert_eq!(fb[0], Feedback::Gray); // Excess E (only 3 in solution, 2 matched green, 1 matched yellow)
    }

    #[test]
    fn test_entropy_logic_remnants() {
        // Logic check: entropy on 1 word should be 0.
        let pool = vec![to_word("ABCDE")];
        let e = calculate_entropy(&to_word("ABCDE"), &pool);
        assert_eq!(e, 0.0);
    }

    // --- B. Input Parsing Tests ---

    #[test]
    fn test_parse_mask_mode() {
        let guess = "TEST";
        let fb = parse_feedback("g y - g", guess, 4).unwrap();
        assert_eq!(fb[0], Feedback::Green);
        assert_eq!(fb[1], Feedback::Yellow);
        assert_eq!(fb[2], Feedback::Gray);
        assert_eq!(fb[3], Feedback::Green);
    }

    #[test]
    fn test_parse_letter_mode_hybrid() {
        let guess = "TRACE";
        // T(Green) r(Yellow) -(Gray) -(Gray) E(Green)
        let fb = parse_feedback("T r - - E", guess, 5).unwrap();
        assert_eq!(fb[0], Feedback::Green);
        assert_eq!(fb[1], Feedback::Yellow);
        assert_eq!(fb[2], Feedback::Gray);
        assert_eq!(fb[3], Feedback::Gray);
        assert_eq!(fb[4], Feedback::Green);
    }

    #[test]
    fn test_buggy_dingy_conflict() {
        // "BUGGY" has G/Y letters. Input "- - y - g" uses G/Y codes.
        // Heuristic must choose Mask Mode, not Letter Mode (mismatch).
        let guess = "BUGGY";
        let input = "- - y - g";

        let fb = parse_feedback(input, guess, 5).unwrap();

        assert_eq!(
            fb,
            vec![
                Feedback::Gray,
                Feedback::Gray,
                Feedback::Yellow,
                Feedback::Gray,
                Feedback::Green
            ]
        );
    }

    #[test]
    fn test_sanitize_user_guess() {
        assert!(sanitize_user_guess("hello", 5).is_some());
        assert!(sanitize_user_guess("hey", 3).is_some());
        assert!(sanitize_user_guess("hello", 3).is_none()); // Length mismatch
        assert!(sanitize_user_guess("he1lo", 5).is_none()); // Number
    }

    // --- C. File I/O & Cache Hardening ---

    // Creates a temporary file for IO testing using a timestamp for uniqueness
    fn with_temp_file<F>(content: &str, test_fn: F)
    where
        F: FnOnce(&str),
    {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = format!("temp_test_{id}.txt");
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(content.as_bytes()).unwrap();
        }
        test_fn(&path);
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(format!("{path}.cache"));
    }

    #[test]
    fn test_dictionary_loading_valid() {
        with_temp_file("apple\nbeast\ncrate", |path| {
            let (words, len) = load_raw_dictionary(path).unwrap();
            assert_eq!(len, 5);
            assert_eq!(words.len(), 3);
        });
    }

    #[test]
    fn test_dictionary_loading_dirty() {
        // "go" (too short), "banana" (too long), "123" (numbers) are skipped
        with_temp_file("apple\ngo\nbanana\n12345\ncrane", |path| {
            let (words, len) = load_raw_dictionary(path).unwrap();
            assert_eq!(len, 5);
            assert_eq!(words.len(), 2);
        });
    }

    #[test]
    fn test_cache_corruption_resilience() {
        with_temp_file("abc\ndef", |path| {
            let (words, _) = load_raw_dictionary(path).unwrap();
            let cache_path = format!("{path}.cache");
            {
                let mut f = File::create(&cache_path).unwrap();
                f.write_all(b"GARBAGE").unwrap();
            }
            // Should detect corruption and recalculate seamlessly
            let sugg = get_initial_suggestions(path, &words, false);
            assert!(!sugg.is_empty());
        });
    }
}
