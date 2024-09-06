
pub const UPARROW: char = match std::char::from_u32(0x00002191) {
    Some(v) => v,
    None => panic!("Invalid unicode character uparrow"),
};
pub const DOWNARROW: char = match std::char::from_u32(0x00002193) {
    Some(v) => v,
    None => panic!("Invalid unicode character downarrow"),
};
