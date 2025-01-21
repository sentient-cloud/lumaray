//! A couple of functions used to constrain the value
//! of a constant generic argument at compile time.
//!
//! These are supposed to be used like:
//!
//! ```rust
//! struct Foo<const N: usize>
//! where
//!     [(); is_zero_or_one(N) - 1]:,
//! {
//!    // ...
//! }
//!
//! ```
//!
//! This is a workaround for the lack of booleans in const generics, or
//! however you would put that.

pub const fn is_zero_or_one(val: usize) -> usize {
    if val == 0 || val == 1 {
        1
    } else {
        0
    }
}
