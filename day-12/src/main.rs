use std::cmp::min;
use std::fs::File;
use std::io::{self, BufReader, BufRead};
use std::env;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let input = &args[1];

    let input = read_input(input)?;

    Ok(())
}


fn read_input(filename: &String) ->  io::Result<Vec<String>> {
    let file_in = File::open(filename)?;
    Ok(BufReader::new(file_in).lines().map(|x| x.unwrap()).collect())
}
