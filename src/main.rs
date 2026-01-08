use std::collections::HashMap;
use std::io::{self, Write};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Feedback {
    Green,
    Yellow,
    Gray,
}

type Word = Vec<char>;

fn calculate_feedback(guess: &[char], solution: &[char]) -> Vec<Feedback> {
    let len = guess.len();
    let mut feedback = vec![Feedback::Gray; len];
    let mut solution_counts = HashMap::new();

    for &c in solution { *solution_counts.entry(c).or_insert(0) += 1; }

    for i in 0..len {
        if guess[i] == solution[i] {
            feedback[i] = Feedback::Green;
            if let Some(count) = solution_counts.get_mut(&guess[i]) { *count -= 1; }
        }
    }

    for i in 0..len {
        if feedback[i] == Feedback::Green { continue; }
        if let Some(count) = solution_counts.get_mut(&guess[i]) {
            if *count > 0 {
                feedback[i] = Feedback::Yellow;
                *count -= 1;
            }
        }
    }
    feedback
}

fn calculate_entropy(guess: &[char], possible_solutions: &[Word]) -> f64 {
    let mut pattern_counts: HashMap<Vec<Feedback>, usize> = HashMap::new();
    for solution in possible_solutions {
        let pattern = calculate_feedback(guess, solution);
        *pattern_counts.entry(pattern).or_insert(0) += 1;
    }
    let total = possible_solutions.len() as f64;
    pattern_counts.values().map(|&count| {
        let p = count as f64 / total;
        if p > 0.0 { -p * p.log2() } else { 0.0 }
    }).sum()
}

fn main() {
    println!("Core logic implemented. Run tests to verify entropy calculation.");
}
