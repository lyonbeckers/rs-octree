use nalgebra::{Scalar, Vector3};
use num::NumCast;
use serde::{Deserialize, Serialize};

use crate::{agnostic_math::vector_abs, NumTraits};

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone, Copy)]
pub struct Aabb<F: Scalar> {
    pub center: Vector3<F>,
    pub dimensions: Vector3<F>,
}

impl<F: NumTraits + Copy + Clone> Aabb<F> {
    pub fn new(center: Vector3<F>, dimensions: Vector3<F>) -> Self {
        Self { center, dimensions }
    }

    pub fn from_extents(min: Vector3<F>, max: Vector3<F>) -> Self {
        let two: F = NumCast::from(2).unwrap();
        let one: F = NumCast::from(1).unwrap();
        let zero: F = NumCast::from(0).unwrap();

        let mut dimensions = Vector3::new(max.x - min.x, max.y - min.y, max.z - min.z);

        //hacky way to check if F is int, since max is inclusive
        if one / two == zero {
            dimensions.x += one;
            dimensions.y += one;
            dimensions.z += one;
        }

        let center = Vector3::new(
            min.x + dimensions.x / two,
            min.y + dimensions.y / two,
            min.z + dimensions.z / two,
        );

        Self { center, dimensions }
    }

    pub fn get_min(&self) -> Vector3<F> {
        let dimensions = vector_abs(self.dimensions);
        let two: F = NumCast::from(2).unwrap();
        Vector3::new(
            self.center.x - dimensions.x / two,
            self.center.y - dimensions.y / two,
            self.center.z - dimensions.z / two,
        )
    }

    pub fn get_max(&self) -> Vector3<F> {
        let dimensions = vector_abs(self.dimensions);
        let two: F = NumCast::from(2).unwrap();
        let one: F = NumCast::from(1).unwrap();
        let zero: F = NumCast::from(0).unwrap();
        let min = self.get_min();

        let mut max = Vector3::new(
            min.x + dimensions.x,
            min.y + dimensions.y,
            min.z + dimensions.z,
        );

        //hacky way to check if F is int, since max is inclusive
        if one / two == zero {
            max.x -= one;
            max.y -= one;
            max.z -= one;
        }

        max
    }

    fn get_corners<T: nalgebra::SimdRealField + NumCast>(&self) -> [Vector3<T>; 8] {
        let min = self.get_min();
        let max = self.get_max();

        let min_x = NumCast::from(min.x).unwrap();
        let min_y = NumCast::from(min.y).unwrap();
        let min_z = NumCast::from(min.z).unwrap();

        let max_x = NumCast::from(max.x).unwrap();
        let max_y = NumCast::from(max.y).unwrap();
        let max_z = NumCast::from(max.z).unwrap();

        [
            nalgebra::Vector3::new(min_x, min_y, min_z),
            nalgebra::Vector3::new(min_x, max_y, min_z),
            nalgebra::Vector3::new(min_x, min_y, max_z),
            nalgebra::Vector3::new(min_x, max_y, max_z),
            nalgebra::Vector3::new(max_x, min_y, min_z),
            nalgebra::Vector3::new(max_x, max_y, min_z),
            nalgebra::Vector3::new(max_x, min_y, max_z),
            nalgebra::Vector3::new(max_x, max_y, max_z),
        ]
    }

    pub fn rotate<T: nalgebra::SimdRealField + NumCast>(
        &self,
        rotation: nalgebra::Rotation3<T>,
    ) -> Self {
        let corners = self.get_corners();

        let rotated_corners = corners
            .iter()
            .map(|corner| rotation * *corner)
            .collect::<Vec<Vector3<T>>>();
        let mut rotated_corners_iter = rotated_corners.iter();

        if let Some(corner) = rotated_corners_iter.next() {
            let corner_x: F = NumCast::from(corner.x).unwrap();
            let corner_y: F = NumCast::from(corner.y).unwrap();
            let corner_z: F = NumCast::from(corner.z).unwrap();

            let mut min_x: F = corner_x;
            let mut min_y: F = corner_y;
            let mut min_z: F = corner_z;

            let mut max_x: F = corner_x;
            let mut max_y: F = corner_y;
            let mut max_z: F = corner_z;

            for corner in rotated_corners_iter {
                let corner_x: F = NumCast::from(corner.x).unwrap();
                let corner_y: F = NumCast::from(corner.y).unwrap();
                let corner_z: F = NumCast::from(corner.z).unwrap();

                min_x = min_x.min(corner_x);
                min_y = min_y.min(corner_y);
                min_z = min_z.min(corner_z);

                max_x = max_x.max(corner_x);
                max_y = max_y.max(corner_y);
                max_z = max_z.max(corner_z);
            }

            let min = Vector3::<F>::new(min_x, min_y, min_z);

            let max = Vector3::<F>::new(max_x, max_y, max_z);

            let aabb = Aabb::from_extents(min, max);
            return Aabb::new(
                Vector3::zeros(),
                Vector3::new(
                    aabb.dimensions.x.abs(),
                    aabb.dimensions.y.abs(),
                    aabb.dimensions.z.abs(),
                ),
            );
        }

        Aabb::from_extents(Vector3::zeros(), Vector3::zeros())
    }

    pub fn get_intersection(&self, other: Aabb<F>) -> Aabb<F> {
        let min = self.get_min();
        let max = self.get_max();

        let other_min = other.get_min();
        let other_max = other.get_max();

        let intersect_min = Vector3::<F>::new(
            min.x.max(other_min.x),
            min.y.max(other_min.y),
            min.z.max(other_min.z),
        );

        let intersect_max = Vector3::<F>::new(
            max.x.min(other_max.x),
            max.y.min(other_max.y),
            max.z.min(other_max.z),
        );

        Aabb::from_extents(intersect_min, intersect_max)
    }

    pub fn intersects_bounds(&self, other: Aabb<F>) -> bool {
        let min = self.get_min();
        let max = self.get_max();

        let other_min = other.get_min();
        let other_max = other.get_max();

        min.x <= other_max.x
            && max.x >= other_min.x
            && min.y <= other_max.y
            && max.y >= other_min.y
            && min.z <= other_max.z
            && max.z >= other_min.z
    }

    pub fn contains_point(&self, point: Vector3<F>) -> bool {
        let min = self.get_min();
        let max = self.get_max();

        point.x >= min.x
            && point.x <= max.x
            && point.y >= min.y
            && point.y <= max.y
            && point.z >= min.z
            && point.z <= max.z
    }
}
