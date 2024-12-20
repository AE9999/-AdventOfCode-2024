use std::cmp::{min, Ordering};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufReader, BufRead};
use std::env;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let input = &args[1];

    let problem = read_input(input)?;
    solve(&problem, 2, 100);
    Ok(())
}

fn solve(problem: &Problem, distance_cheat: usize, minimal_improvement: usize)  {

    let start_pos = problem.start_point();

    let end_pos = problem.end_point();

    let start_state = State::new(start_pos.clone(),
                                 0,
                                        None,
                                        start_pos.clone().distance(&end_pos));

    let mut to_do: BinaryHeap<State> = BinaryHeap::new();
    to_do.push(start_state);

    let mut local_ubs: HashMap<(Point, Cheat), usize> =  HashMap::new();
    let mut global_lbs: HashMap<Cheat, usize> =  HashMap::new();

    while let Some(state) = to_do.pop() {

        let location = state.position.clone();
        let path_len = state.length.clone();
        let mut global_lb_mine = global_lbs.get(&state.cheated).unwrap_or(&usize::MAX);

        if location == end_pos {
            global_lb_mine = min(global_lb_mine, &path_len);
            // println!("Found a global lb. Key:{:?}, value: {}", state.cheated, global_lb_mine);
            global_lbs.insert(state.cheated.clone(), *global_lb_mine);
            continue;
        }

        for &direction in &[Direction::North, Direction::South, Direction::East, Direction::West] {
            explore_if_feasible(&state,
                                &end_pos,
                                &direction,
                                &distance_cheat,
                                &minimal_improvement,
                                &mut local_ubs,
                                &mut global_lbs,
                                &problem)
                .map(|new_states|
                    new_states.into_iter().for_each(
                        |new_state| to_do.push(new_state)));
        }
    }

    let normal_length = global_lbs.get(&None).unwrap();

    let res = global_lbs.values()
                              .filter(|ub| *ub + 100 <= *normal_length)
                              .count();


    println!("How many cheats would save you at least 100 picoseconds? {}", res);
}

fn explore_if_feasible(state: &State,
                       end_pos: &Point,
                       direction: &Direction,
                       distance_cheat: &usize,
                       minimal_improvement: &usize,
                       local_ubs: &mut HashMap<(Point, Cheat), usize>,
                       global_lbs: &mut HashMap<Cheat, usize>,
                       problem: &Problem) -> Option<Vec<State>> {
    let current_position = state.position.clone();
    let next_pos = current_position.add(&direction.to_dx_dy());

    let states_to_check =
        if problem.get_char_on_point(&next_pos) == Some('#') {
            if state.cheated.is_some() {
                return None
            }
            problem.find_all_points_within_distance_of_point(&next_pos, distance_cheat -1)
                   .iter()
                   .map(|endpoint| {

                      let cheated = Some((next_pos.clone(), endpoint.clone()));
                      let distance = endpoint.distance(&end_pos);
                      State::new(endpoint.clone(),
                                 state.length + current_position.distance(endpoint),
                                 cheated,
                                 distance)
                   })
                  .collect()
        } else {
            let next_distance = end_pos.distance(&next_pos);
            let next_state = State::new(next_pos.clone(),
                                         state.length + 1,
                                              state.cheated.clone(),
                                              next_distance);
            vec!(next_state)
        };

    let mut next_states: Vec<State> = Vec::new();

    for state in states_to_check {

        let has_cheated = state.cheated.is_some();

        let global_lb_mine = global_lbs.get(&state.cheated).unwrap_or(&usize::MAX);

        let current_ub = state.length + state.distance;

        let global_lb_no_cheat = global_lbs.get(&None).unwrap_or(&usize::MAX);
        if  current_ub > *global_lb_mine
            || (has_cheated && (current_ub + minimal_improvement > *global_lb_no_cheat)) {
            continue;
        };

        let my_key = (next_pos.clone(), state.cheated.clone());
        let no_secret_key = (next_pos.clone(), None);
        let local_ub_mine = local_ubs.get(&my_key).unwrap_or(&usize::MAX);
        let local_ub_no_secret = local_ubs.get(&no_secret_key).unwrap_or(&usize::MAX);

        if &state.length > local_ub_mine
           || (has_cheated && &state.length + minimal_improvement > *local_ub_no_secret) {
            continue;
        }

        local_ubs.insert(my_key, state.length.clone());
        next_states.push(state.clone());
    }

    Some(next_states)
}

#[derive(Clone)]
struct Problem {
    map: Vec<Vec<char>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct State {
    position: Point,
    length: usize,
    cheated: Option<(Point, Point)>,
    distance: usize,
}

impl State {
    fn new(position: Point,
           length: usize,
           cheated: Option<(Point, Point)>,
           distance: usize) -> Self {
        Self { position, length, cheated, distance }
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


    fn find_all_points_within_distance_of_point(&self, point: &Point, max_distance: usize) -> Vec<Point> {
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

        // println!("point: {:?} => {:?} ..", point, r);

        r
    }

}

type Cheat = Option<(Point, Point)>;

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
        match (self.cheated.is_none(), other.cheated.is_none()) {
            (true, false) => Ordering::Greater,  // `self` has higher priority (cheated is None)
            (false, true) => Ordering::Less,    // `other` has higher priority (cheated is None)
            _ => {
                let l = self.cheated.clone();
                let r = self.cheated.clone();
                // If both are None or both are Some, compare the `cheated` values if Some
                if l.is_none() && r.is_none() {
                    other.distance.cmp(&self.distance)
                }  else {
                    match l.unwrap().cmp(&r.unwrap()) {
                        Ordering::Equal => other.distance.cmp(&self.distance), // Fall back to distance
                        other_cmp => other_cmp, // Use the comparison result of cheated
                    }
                }
            }
        }
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
