extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn generate_bitmasks_from_hoppings(_item: TokenStream) -> TokenStream {
    "pub const HOP_BITMASKS: [u8; SIZE/2] = [
        0b11110000,
        0b11000000
    ];".parse().unwrap()
}
