#![allow(unused_variables)]

use cutile::error::Error;
use cutile::prelude::*;

#[cutile::module]
mod hello_world_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn hello_world_kernel() {
        let pid0 = program_id(0);
        let pid1 = program_id(1);
        let pid2 = program_id(2);
        let n0 = num_programs(0);
        let n1 = num_programs(1);
        let n2 = num_programs(2);
        cuda_tile_print!(
            "Hello from program <{}, {}, {}> in a grid of <{}, {}, {}> programs!\n",
            pid0, pid1, pid2,
            n0, n1, n2
        );
    }
}

use hello_world_module::hello_world_kernel;

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let launcher = hello_world_kernel();
    launcher.grid((2, 2, 1)).sync_on(&stream)?;
    Ok(())
}
