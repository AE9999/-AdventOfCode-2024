use std::cmp::{min, Ordering};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufReader, BufRead};
use std::env;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let input = &args[1];

    let problem = read_input(input)?;
    solve(&problem, 2, 1);
    Ok(())
}

fn solve(problem: &Problem, distance_cheat: usize, minimal_improvement: usize)  {
    let shortest_points: HashMap<Point, Option<usize>> =
        problem.iter_points().fold(HashMap::new(), |mut acc, point| {
        acc.insert(point.clone(), solve_without_cheating(problem, &point));
        acc
    });

    solve_with_cheating(problem,
                        &shortest_points,
                        distance_cheat,
                        minimal_improvement);
}

fn solve_without_cheating(problem: &Problem, start_pos: &Point) -> Option<usize> {

    let end_pos = problem.end_point();

    if problem.get_char_on_point(start_pos) == Some('#') {
        return None
    }

    let start_state = State::new(start_pos.clone(),
                                 0,
                                 start_pos.clone().distance(&end_pos));

    let mut to_do: BinaryHeap<State> = BinaryHeap::new();
    to_do.push(start_state);

    let mut lb = None;
    let mut ubs: HashMap<Point, usize> =  HashMap::new();

    while let Some(state) = to_do.pop() {
        let location = state.position.clone();
        let path_len = state.length.clone();

        if location == end_pos {
            if lb.is_none() {
                lb = Some(path_len);
            }
            lb = Some(min(path_len, lb.unwrap()));
            continue;
        }
        let directions = [Direction::North, Direction::South, Direction::East, Direction::West];
        let next_states =
            directions.iter()
                      .map(|direction| {
                          let next_pos = location.add(&direction.to_dx_dy());
                          State::new(next_pos.clone(),
                              state.length + 1,
                                     next_pos.distance(&end_pos))
                      })
                      .filter(|state| {
                                  problem.is_on_map(&state.position)
                                      && problem.get_char_on_point(&state.position) != Some('#')
                                      && &state.length < ubs.get(&state.position).unwrap_or(&usize::MAX)
                      })
                     .collect::<Vec<State>>();

       next_states.into_iter().for_each(|state| {
           ubs.insert(state.position.clone(), state.length.clone());
           to_do.push(state);
       })
    }

    lb
}

fn solve_with_cheating(problem: &Problem,
                       pre_calculated_best_routes: &HashMap<Point, Option<usize>>,
                       distance_cheat: usize,
                       minimal_improvement: usize)  {
    let start_pos = problem.start_point();

    let end_pos = problem.end_point();

    let start_state = State::new(start_pos.clone(),
                                 0,
                                 start_pos.clone().distance(&end_pos));

    let base_cost = pre_calculated_best_routes.get(&start_pos).unwrap().unwrap();

    let mut to_do: BinaryHeap<State> = BinaryHeap::new();
    to_do.push(start_state);

    let mut effective_cheats : HashMap<Cheat, usize>= HashMap::new();

    let mut lb = None;
    let mut ubs: HashMap<Point, usize> =  HashMap::new();

    while let Some(state) = to_do.pop() {
        let location = state.position.clone();
        let path_len = state.length.clone();

        if location == end_pos {
            if lb.is_none() {
                lb = Some(path_len);
            }
            lb = Some(min(path_len, lb.unwrap()));
            continue;
        }
        let directions = [Direction::North, Direction::South, Direction::East, Direction::West];

        let next_states =
            directions.iter()
                .map(|direction| {
                    let next_pos = location.add(&direction.to_dx_dy());
                    State::new(next_pos.clone(),
                               state.length + 1,
                               next_pos.distance(&end_pos))
                })
                .filter(|state| {
                    problem.is_on_map(&state.position)
                        && problem.get_char_on_point(&state.position) != Some('#')
                        && &state.length < ubs.get(&state.position).unwrap_or(&usize::MAX)
                })
                .collect::<Vec<State>>();

        next_states.into_iter().for_each(|state| {
            ubs.insert(state.position.clone(), state.length.clone());
            to_do.push(state);
        });

        let opportunities_for_cheating_exits =
            directions.iter()
                      .map(|direction| location.add(&direction.to_dx_dy()))
                      .filter(|point| problem.get_char_on_point(point) == Some('#'))
                      .flat_map(|point| {
                            problem.find_all_cheats_from_a_point(&point, distance_cheat)
                                .into_iter()
                                .map(move |p2| (point.clone(), p2))
                       })
                      .collect::<Vec<Cheat>>();

        for (start, exit) in opportunities_for_cheating_exits {

            let cost =
                path_len + 1 + start.distance(&exit) + pre_calculated_best_routes.get(&exit).unwrap().unwrap();
            if cost + minimal_improvement <= base_cost {
                println!("found a cheat for {:?}", (&start, &exit));
                effective_cheats.insert((start, exit), base_cost - cost);
            }
        }
    }

    let x = effective_cheats.values().map(|v|*v).collect::<Vec<usize>>();

    let mut counts = HashMap::new();
    for value in x {
        *counts.entry(value).or_insert(0) += 1;
    }

    // Collect the counts into a vector and sort by value (key)
    let mut sorted_counts: Vec<(usize, usize)> = counts.into_iter().collect();
    sorted_counts.sort_by_key(|&(value, _)| value);

    // Print the counts in ascending order
    for (value, count) in sorted_counts {
        println!("There are {} cheats that save {} picoseconds", count, value);
    }

    let res = effective_cheats.len();
    println!("How many cheats would save you at least 100 picoseconds? {}", res);
}

#[derive(Clone)]
struct Problem {
    map: Vec<Vec<char>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct State {
    position: Point,
    length: usize,
    distance: usize,
}

impl State {
    fn new(position: Point,
           length: usize,
           distance: usize) -> Self {
        Self { position, length, distance }
    }
}

impl Problem {
    fn new(map: Vec<Vec<char>>) -> Self {
        Problem { map}
    }

    fn width(&self) -> i32 {
        self.map.first().unwrap().len() as i32
    }

    fn height(&self) -> i32 {
        self.map.len() as i32
    }
    fn iter_points(&self) -> impl Iterator<Item = Point> + '_ {
        (0..self.height()).flat_map(move |y| {
            (0..self.width()).map(move |x| Point::new(x, y))
        })
    }

    fn is_on_map(&self, point: &Point) -> bool {
        point.x < self.width()  && point.x >= 0  && point.y >= 0 && point.y < self.height()
    }

    fn get_char_on_point(&self, point: &Point) -> Option<char> {
        if self.is_on_map(point) {
            Some(self.map[point.y as usize][point.x as usize])
        } else {
            None
        }
    }

    fn start_point (&self) -> Point {
        self.iter_points().find(|p| self.get_char_on_point(p) == Some('S')).unwrap()
    }

    fn end_point (&self) -> Point {
        self.iter_points().find(|p| self.get_char_on_point(p) == Some('E')).unwrap()
    }


    fn find_all_cheats_from_a_point(&self, point: &Point, max_distance: usize) -> Vec<Point> {
        let mut results = Vec::new();

        // Up and Right
        for dx in 0..=max_distance as i32 {
            for dy in 0..=(max_distance as i32 - dx) {
                results.push(Point::new(point.x + dx, point.y + dy));
            }
        }

        // Up and Left
        for dx in 0..=max_distance as i32 {
            for dy in 0..=(max_distance as i32 - dx) {
                results.push(Point::new(point.x - dx, point.y + dy));
            }
        }

        // Down and Left
        for dx in 0..=max_distance as i32 {
            for dy in 0..=(max_distance as i32 - dx) {
                results.push(Point::new(point.x - dx, point.y - dy));
            }
        }

        // Down and Right
        for dx in 0..=max_distance as i32 {
            for dy in 0..=(max_distance as i32 - dx) {
                results.push(Point::new(point.x + dx, point.y - dy));
            }
        }

        let r: HashSet<Point> =
            results.into_iter().filter(|p|
                p != point
                    && self.is_on_map(p)
                    && self.get_char_on_point(p) != Some('#')).collect();

        let r = r.into_iter().collect();
        println!("point: {:?} => {:?}", point, r);

        r
    }

}

type Cheat = (Point, Point);

#[derive(Hash, Eq, PartialEq, Debug, Clone, PartialOrd, Ord)]
struct Point {
    x: i32,
    y: i32
}

impl Point {
    fn new(x: i32, y: i32) -> Self {
        Point { x, y }
    }

    fn add(&self, other: &Point) -> Self {
        Point { x: self.x + other.x, y: self.y + other.y  }
    }

    fn distance(&self, other: &Point) -> usize {
        (self.x.abs_diff(other.x) + self.y.abs_diff(other.y)) as usize
    }
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // First, compare the `cheated` field
        other.distance.cmp(&self.distance)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum Direction {
    North,
    South,
    East,
    West,
}

impl Direction {
    fn to_dx_dy(&self) -> Point {
        match &self {
            Direction::North => Point::new(0, 1),
            Direction::South => Point::new(0, -1),
            Direction::East => Point::new(1, 0),
            Direction::West => Point::new(-1, 0),
        }
    }
}


fn read_input(filename: &String) ->  io::Result<Problem> {
    let file_in = File::open(filename)?;
    let map = BufReader::new(file_in)
        .lines()
        .map(|line| line.unwrap()
            .chars()
            .map(|c|c.to_string().parse::<char>().unwrap())
            .collect::<Vec<char>>())
        .collect::<Vec<Vec<char>>>();
    Ok(Problem::new(map))
}
