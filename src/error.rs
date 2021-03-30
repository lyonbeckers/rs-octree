use nalgebra::Scalar;
use thiserror::Error;

use crate::{InsertionError, SubdivisionError};

#[derive(Error, Debug)]
pub enum Error<N: Scalar> {
    #[error("insertion error: {0}")]
    InsertionError(#[from] InsertionError<N>),

    #[error("insertion error: {0}")]
    SubdivisionError(#[from] SubdivisionError<N>),
}

pub type Result<T, N> = std::result::Result<T, Error<N>>;
