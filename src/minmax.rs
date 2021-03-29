pub trait MinMax {
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
}

impl MinMax for i8 {
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
