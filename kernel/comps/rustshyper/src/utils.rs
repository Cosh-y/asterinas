use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::str;

pub fn format_binary64_grouped(val: u64) -> String {
    let bits = format!("{:064b}", val);

    let grouped = bits.as_bytes()
        .chunks(4)
        .map(|chunk| str::from_utf8(chunk).unwrap())
        .collect::<Vec<_>>()
        .join("_");
    
    format!("0b{}", grouped)
}