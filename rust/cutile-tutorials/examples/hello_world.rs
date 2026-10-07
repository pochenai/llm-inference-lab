#![allow(unused_variables)]

/// tile programming划分的是执行单元，只不过通过program_id可以决定load那一部分数据(tiles data)从而进行切分
/// grid 划分的是执行单元（Tile Block 的数量与维度）; 具体每个tile block有多少threads由编译器决定
/// program_id 是连接执行单元与数据切片的桥梁;
/// 数据怎么切、加载哪一部分、甚至是重复加载，完全由你在 Kernel 函数里写的内存读取代码（如 load_tile）说了算
use cutile::error::Error;
use cutile::prelude::*;

#[cutile::module]
mod hello_world_module {
    use cutile::core::*;

    #[cutile::entry()]
    /// 每个tile block都会执行相同的程序，他们都加载相同的data(空data)
    fn hello_world_kernel() {
        let pid0 = program_id(0);
        let pid1 = program_id(1);
        let pid2 = program_id(2);
        let n0 = num_programs(0);
        let n1 = num_programs(1);
        let n2 = num_programs(2);
        cuda_tile_print!(
            "Hello from program <{}, {}, {}> in a grid of <{}, {}, {}> programs!\n",
            pid0,
            pid1,
            pid2,
            n0,
            n1,
            n2
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
