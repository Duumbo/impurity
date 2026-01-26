// Problem when compiling with N = ARRAY_SIZE, one array is out of bound,
// Should look into this, seems like a waste to use u32 for 16 Sites for example
/// TODOC
pub fn generate_bitmask(transfer_matrix: &[f64], size: usize) -> Vec<SpinState> {
    const WORD_SIZE: usize = 8;
    let all_zeros = SpinState{state: [0x00; ARRAY_SIZE]};
    let mut hop_tmp: Vec<SpinState> = Vec::with_capacity(size / 2);
    // Index for array
    let mut i: usize = 0;
    while i < size / 2 {
        //println!("i = {i}");
        hop_tmp.push({
            let mut mask = all_zeros;
            let one: u8 = 1;
            let mut j: usize = 0;
            while j < size {
                //println!("j = {j}");
                //println!("t[j + i + 1, j] = {}", transfer_matrix[(j + i + 1)%size + size * j]);
                if transfer_matrix[(j + i + 1)%size + size * j] != 0.0 {
                    if i == 0 {
                        //println!("j + 2 / ws = {}", (j + 2) / WORD_SIZE);
                        mask.state[(j + 1) / WORD_SIZE] ^= one << (WORD_SIZE - (j + 2) % WORD_SIZE);
                    } else {
                        let a = i + j + 2;
                        let s = a % size;
                        let u = s % WORD_SIZE;
                        let landing_hop = (j + i + 1) % size;
                        let byte_idx = landing_hop / WORD_SIZE;
                        let bit_idx = landing_hop % WORD_SIZE;
                        //println!("a = {a}, s = {s}, u = {u}");
                        //println!("{j} => {}", (j+i+1)%size);
                        //println!("Byte {byte_idx}, bit {bit_idx}");
                        //println!("Set {}", ((j+i+2)%size) % WORD_SIZE);
                        //println!("In {}", ((j+i+1)%size) / WORD_SIZE);
                        //println!("Bm {:08b}",one << (WORD_SIZE - ((j+i+2)%size) % WORD_SIZE));
                        mask.state[byte_idx] ^= one << (WORD_SIZE - bit_idx - 1);
                    }
                }
                j += 1;
            }
            if i == 0 && transfer_matrix[size - 1] != 0.0 {
                mask.state[0] ^= one << (WORD_SIZE - 1);
            }
            // If last bitmask, we need to keep only half
            if i == (size / 2) - 1 {
                let mut j: usize = size / 2;
                while j < size {
                    mask.state[j / WORD_SIZE] &= !(one << (WORD_SIZE - j - 1 % WORD_SIZE));
                    j += 1;
                }
            }
            // Index for  HOPPINGS
            //println!("{}", mask);
            mask
        });
        i += 1;
    }
    hop_tmp
}
