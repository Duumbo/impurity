pub fn generate_bitmask(transfer_matrix: &[f64], size: usize) -> Vec<SpinState> {
    const WORD_SIZE: usize = 8;
    let all_zeros = SpinState{state: [0x00; ARRAY_SIZE]};
    let mut hop_tmp: Vec<SpinState> = Vec::with_capacity(size / 2);
    // Index for array
    let mut i: usize = 0;
    while i < size / 2 {
        hop_tmp.push({
            let mut mask = all_zeros;
            let one: u8 = 1;
            let mut j: usize = 0;
            while j < (size - 1 - i) {
                if transfer_matrix[j + i + 1 + SIZE * j] != 0.0 {
                    if i == 0 {
                        mask.state[(j + 2) / WORD_SIZE] ^= one << (WORD_SIZE - (j + 2) % WORD_SIZE);
                    } else {
                        mask.state[(j + 1) / WORD_SIZE] ^= one << (WORD_SIZE - (j + 1) % WORD_SIZE);
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
            mask
        });
        i += 1;
    }
    hop_tmp
}
