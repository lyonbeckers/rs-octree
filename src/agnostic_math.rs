use nalgebra::{Scalar, Vector3};

pub trait MinMax {
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
}

impl MinMax for u8 {
    fn min(self, other: Self) -> Self {
        std::cmp::min(self, other)
    }
    fn max(self, other: Self) -> Self {
        std::cmp::max(self, other)
    }
}

impl MinMax for u16 {
    fn min(self, other: Self) -> Self {
        std::cmp::min(self, other)
    }
    fn max(self, other: Self) -> Self {
        std::cmp::max(self, other)
    }
}

impl MinMax for u32 {
    fn min(self, other: Self) -> Self {
        std::cmp::min(self, other)
    }
    fn max(self, other: Self) -> Self {
        std::cmp::max(self, other)
    }
}

impl MinMax for u64 {
    fn min(self, other: Self) -> Self {
        std::cmp::min(self, other)
    }
    fn max(self, other: Self) -> Self {
        std::cmp::max(self, other)
    }
}

impl MinMax for i8 {
    fn min(self, other: Self) -> Self {
        std::cmp::min(self, other)
    }
    fn max(self, other: Self) -> Self {
        std::cmp::max(self, other)
    }
}

impl MinMax for i16 {
    fn min(self, other: Self) -> Self {
        std::cmp::min(self, other)
    }
    fn max(self, other: Self) -> Self {
        std::cmp::max(self, other)
    }
}

impl MinMax for i32 {
    fn min(self, other: Self) -> Self {
        std::cmp::min(self, other)
    }
    fn max(self, other: Self) -> Self {
        std::cmp::max(self, other)
    }
}

impl MinMax for i64 {
    fn min(self, other: Self) -> Self {
        std::cmp::min(self, other)
    }
    fn max(self, other: Self) -> Self {
        std::cmp::max(self, other)
    }
}

impl MinMax for f32 {
    fn min(self, other: Self) -> Self {
        self.min(other)
    }
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
}

impl MinMax for f64 {
    fn min(self, other: Self) -> Self {
        self.min(other)
    }
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
}

pub trait AgnosticAbs {
    fn abs(self) -> Self;
}

impl AgnosticAbs for i8 {
    fn abs(self) -> Self {
        self.abs()
    }
}

impl AgnosticAbs for i16 {
    fn abs(self) -> Self {
        self.abs()
    }
}

impl AgnosticAbs for i32 {
    fn abs(self) -> Self {
        self.abs()
    }
}

impl AgnosticAbs for i64 {
    fn abs(self) -> Self {
        self.abs()
    }
}

impl AgnosticAbs for f32 {
    fn abs(self) -> Self {
        self.abs()
    }
}

impl AgnosticAbs for f64 {
    fn abs(self) -> Self {
        self.abs()
    }
}

impl AgnosticAbs for u8 {
    fn abs(self) -> Self {
        self
    }
}

impl AgnosticAbs for u16 {
    fn abs(self) -> Self {
        self
    }
}

impl AgnosticAbs for u32 {
    fn abs(self) -> Self {
        self
    }
}

impl AgnosticAbs for u64 {
    fn abs(self) -> Self {
        self
    }
}

pub fn vector_abs<N>(vector: Vector3<N>) -> Vector3<N>
where
    N: Scalar + AgnosticAbs + Copy,
{
    Vector3::<N>::new(vector.x.abs(), vector.y.abs(), vector.z.abs())
}
