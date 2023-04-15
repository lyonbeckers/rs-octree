use std::sync::Arc;

use aabb::Aabb;
use nalgebra::Vector3;
use octree::{Octree, OctreeVec, PointData};
use parking_lot::RwLock;
use tracing::Subscriber;
use tracing_subscriber::EnvFilter;

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct TileData {
    point: Vector3<i32>,
}

impl PointData<i32> for TileData {
    fn get_point(&self) -> Vector3<i32> {
        self.point
    }
}

fn setup_subscriber() -> impl Subscriber {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .finish()
}

pub fn main() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let container = Arc::new(RwLock::new(OctreeVec::<i32, TileData, 32>::new()));
    let octree = Octree::<i32, TileData, 32>::new(
        Aabb::from_extents(Vector3::new(0, 0, 0), Vector3::new(10, 10, 10)),
        None,
        container,
    );

    octree
        .write()
        .insert_elements(vec![TileData {
            point: Vector3::new(0, 0, 0),
        }])
        .ok();
}
